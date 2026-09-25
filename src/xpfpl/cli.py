"""Command line: xpfpl fetch | train | compare | tune | predict | recommend."""

import argparse
import sys

import numpy as np
import pandas as pd

from xpfpl import config, models


def cmd_fetch(args) -> None:
    from xpfpl.data.history import build_matches
    build_matches(refresh=args.refresh)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    err = y_pred - y_true
    return f"RMSE {np.sqrt(np.mean(err ** 2)):.3f}  MAE {np.mean(np.abs(err)):.3f}"


def _benchmarks(val_df: pd.DataFrame, name: str) -> dict[str, np.ndarray]:
    """The comparisons every report carries: the no-ML baseline, and FPL's own xP where recorded."""
    from xpfpl import models, validate

    preds = {}
    if name != "baseline":
        preds[validate.REFERENCE] = models.load("baseline").predict(val_df)
    fpl = val_df.get("fpl_xp_prev")
    if fpl is not None and fpl.notna().mean() > 0.5:
        preds[validate.FPL_XP] = fpl.fillna(0.0).to_numpy(dtype="float32")
    return preds


def _horizon_preds(predictor, stale_frames: dict, name: str) -> dict[int, dict[str, np.ndarray]]:
    """Predictions from features as known k gameweeks before each validation match."""
    from xpfpl import models, validate

    out = {}
    for k, frame in stale_frames.items():
        out[k] = {name: predictor.predict(frame)}
        if name != "baseline":
            out[k][validate.REFERENCE] = models.load("baseline").predict(frame)
    return out


def _validate(predictor, train_df, val_df, name: str = config.MODEL, stale_frames: dict | None = None) -> None:
    """Score the held-out season, print it and save the report the dashboard charts."""
    from xpfpl import validate
    from xpfpl.features import TARGET

    active = val_df["played_r5"] > 0
    y = val_df[TARGET].to_numpy()
    preds = {name: predictor.predict(val_df), **_benchmarks(val_df, name)}
    print(f"\nValidation ({val_df['season'].iloc[0]})          all rows                 active players")
    for label, pred in preds.items():
        print(f"  {label:24s} {_metrics(y, pred)}   {_metrics(y[active], pred[active])}")

    # The ceiling needs a component model's probabilities; `xpfpl compare` computes it.
    comparison = validate.load_comparison() or {}
    ceiling = comparison.get("ceiling") if comparison.get("season") == val_df["season"].iloc[0] else None
    report = validate.build_report(
        val_df, preds, target=TARGET, primary=name,
        trained_on=f"{train_df['season'].min()} to {train_df['season'].max()}",
        best_epoch=predictor.meta.get("best_epoch"),
        horizons=_horizon_preds(predictor, stale_frames, name) if stale_frames else None,
        ceiling=ceiling)
    validate.save_report(report)
    print("\n" + validate.summarise(report))
    print(f"Saved the accuracy report to {config.VALIDATION_PATH}")


def cmd_train(args) -> None:
    from xpfpl import models
    from xpfpl.models.trainer import TrainConfig

    if args.model == "baseline":
        raise SystemExit("The baseline has nothing to train - use it directly with --model baseline.")
    frame, train_df, val_df, stale = _split(args.val_season, horizons=args.horizons)
    print(f"Train: {len(train_df):,} rows ({train_df['season'].min()} to {train_df['season'].max()})"
          f"  Validation: {len(val_df):,} rows ({args.val_season})  Model: {args.model}")

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    predictor = models.fit(args.model, train_df, val_df, cfg=cfg)
    _validate(predictor, train_df, val_df, args.model, stale)

    if not args.no_final:
        # Refit on everything (incl. the validation season and this season) for the best epoch
        # count, so the live model learns the latest scoring rules and form.
        refit = models.refit_config(predictor, lr=args.lr, batch_size=args.batch_size)
        print(f"\nFinal fit on all {len(frame):,} rows for {refit.epochs} epochs...")
        predictor = models.fit(args.model, frame, None, cfg=refit)
    predictor.meta["val_season"] = args.val_season
    predictor.save(models.path(args.model))
    print(f"Saved model to {models.path(args.model)}")
    if args.model != config.MODEL:
        print(f"config.MODEL is still '{config.MODEL}' - set it to '{args.model}' in "
              f"src/xpfpl/config.py to use this model by default.")


def _split(val_season: str, horizons: int = 1):
    """The training frame split into everything before `val_season` and `val_season` itself,
    plus `val_season`'s rows with features as known k = 2..`horizons` gameweeks earlier."""
    from xpfpl.data.history import load_matches
    from xpfpl.features import build_training_frame

    print("Building features...")
    matches = load_matches()
    frame = build_training_frame(matches)
    year = frame["season"].str[:4].astype(int)
    train_df = frame[year < int(val_season[:4])]
    val_df = frame[frame["season"] == val_season]
    if val_df.empty:
        raise SystemExit(f"No rows for {val_season}. Seasons: {sorted(frame['season'].unique())}")
    stale = {}
    for k in range(2, horizons + 1):
        print(f"Building features as known {k} gameweeks ahead...")
        stale_frame = build_training_frame(matches, stale=k)
        stale[k] = stale_frame[stale_frame["season"] == val_season]
    return frame, train_df, val_df, stale


def cmd_validate(args) -> None:
    """Score a held-out season and write the accuracy report, without touching the saved model."""
    from xpfpl import models
    from xpfpl.models.trainer import TrainConfig

    _, train_df, val_df, stale = _split(args.val_season, horizons=args.horizons)
    print(f"Train: {len(train_df):,} rows  Validation: {len(val_df):,} rows ({args.val_season})")
    predictor = models.fit(args.model, train_df, val_df, cfg=TrainConfig(epochs=args.epochs))
    _validate(predictor, train_df, val_df, args.model, stale)


def cmd_compare(args) -> None:
    """Train every model on the same split and score them side by side."""
    import json
    import time
    from datetime import datetime

    from xpfpl import models, validate
    from xpfpl.features import TARGET
    from xpfpl.models.trainer import TrainConfig

    names = args.models or list(models.NAMES)
    _, train_df, val_df, stale = _split(args.val_season, horizons=args.horizons)
    y = val_df[TARGET].to_numpy()
    active = (val_df["played_r5"] > 0).to_numpy()

    preds, rows, horizons, ceiling = {}, [], {k: {} for k in stale}, None
    for name in names:
        print(f"\n--- {name}: {models.DESCRIPTIONS[name]} ---")
        started = time.time()
        predictor = models.fit(name, train_df, val_df, cfg=TrainConfig(epochs=args.epochs))
        pred = predictor.predict(val_df)
        preds[name] = pred
        for k, frame in stale.items():
            horizons[k][name] = predictor.predict(frame)
        if name == "components":
            print("  simulating a perfect model from its probabilities...")
            ceiling = validate.simulate_ceiling(predictor.components(val_df), active,
                                                tables=validate.match_tables(train_df))
            ceiling["actual_variance"] = float(np.var(y[active]))
        rows.append({"model": name, "description": models.DESCRIPTIONS[name],
                     **validate._scores(y[active], pred[active]),
                     "epochs": predictor.meta.get("best_epoch"),
                     "seconds": round(time.time() - started, 1)})
        print(f"  {name}: {_metrics(y[active], pred[active])} in {rows[-1]['seconds']}s")
    for label, pred in _benchmarks(val_df, "baseline").items():
        preds[label] = pred
        rows.append({"model": label, "description": "FPL's own expected points (its value after the previous match)",
                     **validate._scores(y[active], pred[active]), "epochs": None, "seconds": None})

    report = validate.build_report(val_df, preds, target=TARGET, primary=names[0],
                                   trained_on=f"{train_df['season'].min()} to {train_df['season'].max()}",
                                   horizons=horizons or None, ceiling=ceiling)
    captain = pd.DataFrame(report["captain"])
    ranking = pd.DataFrame(report["ranking"]).groupby("model")["spearman"].mean()
    table = pd.DataFrame(rows)
    table["spearman"] = table["model"].map(ranking)
    table["captain_pts_per_gw"] = [captain[n].mean() if n in captain else float("nan") for n in table["model"]]

    print(f"\n--- Models on {args.val_season} (players getting minutes) ---")
    print(table[["model", "rmse", "mae", "r2", "spearman", "captain_pts_per_gw", "epochs", "seconds"]]
          .sort_values("rmse").round(3).to_string(index=False))
    if report["horizons"]:
        hz = pd.DataFrame(report["horizons"]).pivot(index="model", columns="horizon", values="rmse")
        print("\nRMSE by how many gameweeks ahead the forecast was made:\n" + hz.round(3).to_string())
    groups = pd.DataFrame(report["return_groups"])
    if len(groups):
        print("\nRMSE by what actually happened (all rows):\n"
              + groups.pivot(index="model", columns="group", values="rmse").round(3).to_string())
    if ceiling:
        print(f"\nPerfect-model ceiling: RMSE {ceiling['rmse_median']:.3f} "
              f"({ceiling['rmse_p5']:.3f}-{ceiling['rmse_p95']:.3f}), R² {ceiling['r2_median']:.3f}. "
              f"Spread of points: simulated {ceiling['outcome_variance']:.2f}, "
              f"actual {ceiling['actual_variance']:.2f} (if simulated is lower, the ceiling is optimistic)")
    config.COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.COMPARISON_PATH.write_text(json.dumps(
        {"generated": datetime.now().isoformat(timespec="seconds"), "season": args.val_season,
         "rows": int(active.sum()), "models": table.to_dict("records"), "ceiling": ceiling,
         "headline": report["headline"], "horizons": report["horizons"],
         "return_groups": report["return_groups"]}, indent=1),
        encoding="utf-8")
    print(f"\nSaved to {config.COMPARISON_PATH}")


def cmd_tune(args) -> None:
    """Search the optimiser tuning parameters with the backtest and print the config.py block to paste."""
    from xpfpl import tune
    from xpfpl.data.history import load_matches
    from xpfpl.features import build_training_frame

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    print("Building features...")
    frame = build_training_frame(load_matches())
    chip_stages = [] if args.no_chips else tune.CHIP_STAGES
    stages = tune.STAGES
    if args.stages:                       # e.g. --stages "price changes" wildcard
        wanted = [s.lower() for s in args.stages]
        keep = lambda group: [st for st in group if any(w in st[0].lower() for w in wanted)]
        stages, chip_stages = keep(stages), keep(chip_stages)
        if not stages and not chip_stages:
            raise SystemExit("No stage matched. Stages: "
                             + ", ".join(repr(n) for n, _ in tune.STAGES + tune.CHIP_STAGES))
    report = tune.tune(seasons, model=args.model, frame=frame, stages=stages,
                       chip_stages=chip_stages, verbose=args.verbose)
    print("\n" + tune.summarise(report))
    print(f"\nEvery trial is in {config.TUNING_PATH}")


def _parse_grid(values: list[str]) -> dict[str, list]:
    """--sweep horizon=3,5,8 discount=0.8,0.9 -> {"horizon": [3, 5, 8], "discount": [0.8, 0.9]}.

    Each value is cast to the type of that setting's default in backtest.Settings.
    """
    from dataclasses import asdict

    from xpfpl.backtest import Settings
    defaults = asdict(Settings())
    grid = {}
    for item in values:
        key, _, raw = item.partition("=")
        if key not in defaults or key == "thresholds":
            raise SystemExit(f"Can't sweep '{key}'. Try: "
                             f"{', '.join(k for k in sorted(defaults) if k != 'thresholds')}")
        kind = type(defaults[key])
        cast = (lambda v: v.lower() in ("1", "true", "yes")) if kind is bool else kind
        grid[key] = [cast(v) for v in raw.split(",")]
    return grid


def cmd_backtest(args) -> None:
    from xpfpl import backtest
    from xpfpl.data.history import load_matches
    from xpfpl.features import build_training_frame

    base = backtest.Settings(season=args.season, start_gw=args.start_gw, end_gw=args.end_gw,
                             horizon=args.horizon, discount=args.discount, ft_value=args.ft_value,
                             bench_weight=args.bench_weight, max_hits=args.max_hits,
                             budget=args.budget, model=args.model, chips=args.chips,
                             plan_transfers=not args.no_plan, price_weight=args.price_weight,
                             pool_size=args.pool_size)
    print("Building features...")
    frame = build_training_frame(load_matches())

    if args.sweep:
        table = backtest.sweep(base, _parse_grid(args.sweep), frame=frame)
        cols = ["points", "points_per_gw", "hits", "transfers", "captain_points", "bench_points",
                *sorted(_parse_grid(args.sweep))]
        print("\n--- Sweep results (best first) ---")
        print(table[cols].round(2).to_string(index=False))
        print(f"\nPer-gameweek detail in {config.BACKTEST_DIR}")
        return

    print(f"Replaying {base.season} GW{base.start_gw}-{base.end_gw} "
          f"(horizon {base.horizon}, discount {base.discount}, model {base.model}"
          f"{', planning transfers' if base.plan_transfers else ''})")
    result = backtest.run(base, frame=frame)
    s = result.summary
    print(f"\n--- {base.season} ---")
    print(f"  Total points     {s['points']:.0f} over {s['gws']} gameweeks ({s['points_per_gw']:.1f} per GW)")
    print(f"  Transfers        {s['transfers']} ({s['hits']} transfer penalties, -{s['hit_cost']} pts)")
    print(f"  Captain          {s['captain_points']:.0f} pts from the armband "
          f"(counted once; doubled in the total)")
    print(f"  Left on the bench{s['bench_points']:>5.0f} pts")
    print(f"  xP vs actual     predicted {s['xp']:.0f}, scored {s['points'] + s['hit_cost']:.0f} "
          f"({s['xp_error_per_gw']:+.1f} per GW)")
    if s["chips"]:
        print(f"  Chips            {s['chips']}")
    csv, _ = result.save()
    print(f"\nSaved to {csv}")


def _top_by_position(players: pd.DataFrame, gameweeks: list[int], n: int) -> None:
    cols = ["name", "team_name", "price"] + [f"xp_{gw}" for gw in gameweeks] + ["xp_total"]
    for pos, label in config.POSITIONS.items():
        top = players[players["position"] == pos].nlargest(n, "xp_total")[cols]
        print(f"\n{label}")
        print(top.round(2).to_string(index=False))


def cmd_predict(args) -> None:
    from xpfpl.predict import predict_upcoming
    players, gameweeks = predict_upcoming(args.horizon, args.model)
    print(f"xP for GW{gameweeks[0]}-{gameweeks[-1]} ({args.model}); full table in {config.PREDICTIONS_DIR}")
    _top_by_position(players, gameweeks, args.top)

    parts = [c for c in players.columns if c.startswith("from_")]
    if parts:
        print(f"\nWhere GW{gameweeks[0]}'s xP comes from (top {args.top} overall):")
        top = players.nlargest(args.top, "xp_total")
        table = top[["name", "team_name"] + parts].rename(columns=lambda c: c.removeprefix("from_"))
        print(table.round(2).to_string(index=False))


def _label(players: pd.DataFrame, p: int) -> str:
    r = players.loc[p]
    return f"{r['name']} ({r['team_name']}, £{r['price']:.1f})"


def _print_plan(players: pd.DataFrame, plan, gw: int, title: str) -> None:
    col = f"xp_{gw}"
    print(f"\n=== {title} - GW{gw}  formation {plan.formation(players, gw)}  xP {plan.xp[gw]:.1f} ===")
    for p in plan.lineups[gw]:
        tag = " (C)" if p == plan.captains[gw] else " (VC)" if p == plan.vice_captains[gw] else ""
        pos = config.POSITIONS[players.at[p, "position"]]
        print(f"  {pos}  {_label(players, p):38s} xP {players.at[p, col]:4.1f}{tag}")
    print("  Bench (in order):")
    for i, p in enumerate(plan.bench[gw]):
        pos = config.POSITIONS[players.at[p, "position"]]
        print(f"   {i + 1}. {pos}  {_label(players, p):35s} xP {players.at[p, col]:4.1f}")


def cmd_recommend(args) -> None:
    from xpfpl import chips
    from xpfpl.data import api
    from xpfpl.myteam import load_my_team
    from xpfpl.optimise import solve
    from xpfpl.predict import predict_upcoming

    bs, fx = api.bootstrap(), api.fixtures()
    players, gameweeks = predict_upcoming(args.horizon, args.model, bs, fx)
    gw = gameweeks[0]
    print(f"Planning GW{gw} with a {len(gameweeks)}-GW horizon (GW{gameweeks[0]}-{gameweeks[-1]}), model: {args.model}")

    counts = chips.fixture_counts(fx, gameweeks, list(players["team"].unique()))
    names = {t["id"]: t["short_name"] for t in bs["teams"]}
    for g in gameweeks:
        blanks = [names[t] for t in counts.index if counts.at[t, g] == 0]
        doubles = [names[t] for t in counts.index if counts.at[t, g] >= 2]
        if blanks or doubles:
            print(f"  GW{g}: blank {blanks or '-'}  double {doubles or '-'}")

    if args.team_id is None:
        plan = solve(players, gameweeks, bank=args.budget)
        print(f"\nNo --team-id given: best squad from scratch with £{args.budget}m "
              f"(£{plan.budget_left}m left).")
        _print_plan(players, plan, gw, "Best squad")
        return

    me = load_my_team(args.team_id, bs)
    ft = me.free_transfers if args.free_transfers is None else args.free_transfers
    bank = me.bank if args.bank is None else args.bank
    print(f"\nTeam: {me.name}  bank £{bank}m  free transfers {ft}"
          f"{' (estimated - override with --free-transfers)' if args.free_transfers is None else ''}")
    if me.pending_transfers:
        print(f"  Including {me.pending_transfers} transfer(s) you've already made for GW{gw}.")

    kwargs = dict(current_squad=me.squad, bank=bank)
    if args.active_chip:
        # Pending transfers and active chips aren't public before the deadline, so plan the
        # chip squad from the last-deadline squad and selling prices, which is the right baseline.
        name = chips.CHIP_NAMES[args.active_chip]
        horizon = [gw] if args.active_chip == "freehit" else gameweeks
        chip_plan = solve(players, horizon, **kwargs, unlimited_transfers=True)
        print(f"\n--- {name} active for GW{gw} (unlimited free transfers) ---")
        print(f"  Starting from your GW{gw - 1} squad; transfers you've already made aren't visible "
              f"before the deadline, so compare against the list below.")
        kept = [p for p in me.squad if p not in chip_plan.transfers_out]
        print(f"  KEEP: {', '.join(players.at[p, 'name'] for p in kept) or '-'}")
        print(f"  OUT:  {', '.join(players.at[p, 'name'] for p in chip_plan.transfers_out) or '-'}")
        print(f"  IN:   {', '.join(_label(players, p) for p in chip_plan.transfers_in) or '-'}")
        print(f"  Bank after: £{chip_plan.budget_left}m")
        _print_plan(players, chip_plan, gw, f"{name} team")
        print(f"\n--- Chips ---\n  {name} is active: no other chip can be played in GW{gw}.")
        return

    plan_transfers = not args.no_plan
    plan = solve(players, gameweeks, **kwargs, free_transfers=ft, max_hits=args.max_hits,
                 plan_transfers=plan_transfers, price_weight=config.PRICE_WEIGHT,
                 pool_size=config.POOL_SIZE if plan_transfers else None)
    hold = solve(players, gameweeks, **kwargs, free_transfers=0, max_hits=0)
    for note in plan.notes:
        print(f"  Note: {note}")

    print("\n--- Transfers ---")
    if not plan.transfers_in:
        print(f"  Roll the transfer(s) - nothing beats keeping the squad.")
    else:
        for out_id, in_id in zip(sorted(plan.transfers_out, key=lambda p: players.at[p, "position"]),
                                 sorted(plan.transfers_in, key=lambda p: players.at[p, "position"])):
            print(f"  OUT {_label(players, out_id):38s} IN {_label(players, in_id)}")
        gain = sum(plan.xp.values()) - sum(hold.xp.values()) - config.HIT_COST * plan.total_hits
        whole = " for the whole plan" if plan.future_transfers else ""
        print(f"  Transfer penalties: {plan.total_hits} (-{config.HIT_COST * plan.total_hits} pts).  "
             f"Net gain over {len(gameweeks)} GWs{whole}: {gain:+.1f} xP.  "
             f"Bank after: £{plan.budget_left}m")

    if plan.future_transfers:
        print("\n--- The weeks after that (re-planned every week from fresh predictions) ---")
        for future_gw, moves in sorted(plan.future_transfers.items()):
            outs = ", ".join(players.at[p, "name"] for p in moves["out"]) or "-"
            ins = ", ".join(_label(players, p) for p in moves["in"]) or "-"
            cost = f"  (-{config.HIT_COST * moves['hits']} pts)" if moves["hits"] else ""
            print(f"  GW{future_gw}: OUT {outs}   IN {ins}{cost}")

    _print_plan(players, plan, gw, "Your team")

    if args.sensitivity:
        from xpfpl.optimise import sensitivity
        print(f"\n--- How robust is that? ({args.sensitivity} re-solves with every player's xP "
              f"randomly off by ~25%) ---")
        robust = sensitivity(players, gameweeks, sims=args.sensitivity, **kwargs, free_transfers=ft,
                             max_hits=args.max_hits, price_weight=config.PRICE_WEIGHT)
        for name, table in robust.items():
            for row in table.head(5).itertuples():
                print(f"  {name[:-1]:8s} {row.share:4.0%}  {row[1]}")
        print("  A move that wins in most runs is robust; one that rarely wins is a close call.")

    risers = players.loc[plan.squad].nlargest(3, "price_delta") if "price_delta" in players else None
    if risers is not None and float(risers["price_delta"].iloc[0]) > 0.005:
        moves = ", ".join(f"{r['name']} {r['price_delta'] * 10:+.2f}" for _, r in risers.iterrows())
        print(f"\n  Expected price moves in your squad (tenths per GW): {moves}")

    available = chips.available_chips(bs, me.chips_used, gw)
    print("\n--- Chips ---")
    if not available:
        print("  No chips available this gameweek.")
    else:
        advice = chips.advise(players, gameweeks, plan, available, kwargs)
        for a in advice:
            flag = "PLAY" if a.recommended else "save"
            expiry = "  [expires after this GW - use it or lose it]" if a.expiring else ""
            print(f"  {flag:4s} {chips.CHIP_NAMES[a.chip]:15s} gain {round(a.gain, 1) + 0.0:5.1f} / threshold "
                  f"{a.threshold:4.1f}  - {a.reason}{expiry}")
        best = next((a for a in advice if a.recommended), None)
        if best and best.chip in ("freehit", "wildcard"):
            horizon = [gw] if best.chip == "freehit" else gameweeks
            chip_plan = solve(players, horizon, **kwargs, unlimited_transfers=True)
            print(f"\n  {chips.CHIP_NAMES[best.chip]} squad - OUT: "
                  f"{', '.join(players.at[p, 'name'] for p in chip_plan.transfers_out)}")
            print(f"  IN: {', '.join(players.at[p, 'name'] for p in chip_plan.transfers_in)}")
            _print_plan(players, chip_plan, gw, f"{chips.CHIP_NAMES[best.chip]} team")


def cmd_markets(args) -> None:
    """Betting-market odds from Polymarket: every EPL match since 2024-25, priced at its FPL
    deadline, with the expected goals the odds imply; plus today's season-long markets."""
    from xpfpl.data import api, markets
    from xpfpl.data.history import load_matches

    bs = api.bootstrap()
    print("Fetching Polymarket match markets (first run: a few minutes; cached afterwards)...")
    matches, scorers = markets.build(load_matches(), bs, cache=not args.refresh)
    outrights = markets.save_outrights(markets.fetch_outrights())
    print(f"{len(matches)} matches ({matches.groupby('season').size().to_dict()}), "
          f"{len(scorers)} player scorer prices, {outrights['snapshot'].nunique()} outright snapshot(s).")

    upcoming = matches[matches["kickoff"] > pd.Timestamp.now(tz="UTC")].sort_values("kickoff")
    shown = upcoming if len(upcoming) else matches.sort_values("kickoff").tail(10)
    label = "Upcoming" if len(upcoming) else "No upcoming matches listed yet - the latest"
    print(f"\n{label}:")
    table = shown.assign(match=shown["home"].str.replace(" FC", "") + " v " + shown["away"].str.replace(" FC", ""))
    cols = ["gw", "match", "home_win", "draw", "away_win", "lam_home", "lam_away", "over_2.5", "btts"]
    print(table[[c for c in cols if c in table]].round(2).to_string(index=False))


def cmd_archive(args) -> None:
    """Copy everything on disk into archive/ (git-tracked), or take the pre-deadline snapshot."""
    from xpfpl.data import api, archive

    if args.deadline:
        path = archive.save_deadline(api.bootstrap())
        print(f"Saved {path}" if path else "No deadline ahead.")
        return
    archive.backfill(markets_too=not args.no_markets)
    print("\nArchive:\n" + archive.size_report())


def _saved_team_id() -> int | None:
    """The team id the dashboard saved (data/app_settings.json), if any."""
    import json
    path = config.DATA_DIR / "app_settings.json"
    return json.loads(path.read_text(encoding="utf-8")).get("team_id") if path.exists() else None


def cmd_export(args) -> None:
    """Write the website's data (web/public/data/)."""
    from xpfpl import export
    team_id = args.team_id or _saved_team_id()
    files = export.export(team_id=team_id, model=args.model)
    size = sum(f.stat().st_size for f in files)
    print(f"Wrote {len(files)} files ({size / 1e6:.1f} MB) to {config.SITE_DATA_DIR}"
          + ("" if team_id else " - no team id, so no manager.json (use --team-id)"))


def cmd_publish(args) -> None:
    """Export, build the site and push it to the gh-pages branch (GitHub Pages serves it)."""
    from xpfpl import export
    if not args.no_export:
        cmd_export(args)
    dist = export.build_site()
    if args.build_only:
        print(f"Built the site in {dist} (not pushed). Preview it with: npm --prefix web run preview")
        return
    export.publish(dist)
    print("Pushed to the gh-pages branch. GitHub Pages updates a minute or two later.")


def cmd_scorecard(args) -> None:
    """The live record: forecasts saved before each deadline, scored once the GW is played."""
    from xpfpl import scorecard
    from xpfpl.data import api
    from xpfpl.data.history import load_matches

    season = args.season or api.current_season(api.bootstrap())
    report = scorecard.score(load_matches(), season)
    scorecard.save(report)
    print(scorecard.summarise(report))


def cmd_app(args) -> None:
    import subprocess
    import threading
    import webbrowser
    from pathlib import Path
    app = Path(__file__).with_name("app.py")
    url = f"http://localhost:{args.port}"
    print(f"Opening {url} (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(3.0, webbrowser.open, [url]).start()
    # Headless skips Streamlit's first-run email prompt; the browser is opened above instead.
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port),
                    "--server.address", "localhost", "--server.headless", "true",
                    "--browser.gatherUsageStats", "false"])


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # player names on Windows consoles

    parser = argparse.ArgumentParser(prog="xpfpl", description="xP-FPL: Expected Points for FPL")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download past seasons (vaastav) + this season (FPL API)")
    p.add_argument("--refresh", action="store_true", help="re-download files already cached")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("train", help="train an xP model")
    p.add_argument("--model", choices=models.TRAINABLE, default=config.MODEL,
                   help="  ".join(f"{n}: {d}" for n, d in models.DESCRIPTIONS.items() if n != "baseline"))
    p.add_argument("--val-season", default=config.HISTORY_SEASONS[-1])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--no-final", action="store_true",
                   help="save the validation model instead of refitting on all seasons")
    p.add_argument("--horizons", type=int, default=1,
                   help="also score forecasts made up to this many gameweeks ahead (slower)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("validate", help="score a held-out season and write models/validation.json")
    p.add_argument("--model", choices=models.TRAINABLE, default=config.MODEL)
    p.add_argument("--val-season", default=config.HISTORY_SEASONS[-1])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--horizons", type=int, default=3,
                   help="also score forecasts made up to this many gameweeks ahead")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("compare", help="train every model on the same season and rank them")
    p.add_argument("--models", nargs="+", choices=models.NAMES,
                   help="which models to compare (default: all of them)")
    p.add_argument("--val-season", default=config.HISTORY_SEASONS[-1])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--horizons", type=int, default=3,
                   help="also score forecasts made up to this many gameweeks ahead")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("tune", help="search the config.py tuning parameters by replaying whole seasons")
    p.add_argument("--seasons", default=",".join(config.HISTORY_SEASONS[-4:-1]),
                   help="comma-separated seasons to score each candidate on")
    p.add_argument("--model", choices=models.NAMES, default=config.MODEL)
    p.add_argument("--stages", nargs="+", metavar="NAME",
                   help="only run the stages whose name contains one of these (default: all)")
    p.add_argument("--no-chips", action="store_true", help="skip the chip-threshold stages")
    p.add_argument("--verbose", action="store_true", help="print every gameweek of every backtest")
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("backtest", help="replay a past season gameweek by gameweek and count the points")
    p.add_argument("--season", default=config.HISTORY_SEASONS[-2])
    p.add_argument("--start-gw", type=int, default=1)
    p.add_argument("--end-gw", type=int, default=38)
    p.add_argument("--horizon", type=int, default=config.HORIZON)
    p.add_argument("--discount", type=float, default=config.DISCOUNT)
    p.add_argument("--ft-value", type=float, default=config.FT_VALUE)
    p.add_argument("--bench-weight", type=float, default=config.BENCH_WEIGHT)
    p.add_argument("--max-hits", type=int, default=config.MAX_HITS)
    p.add_argument("--budget", type=float, default=100.0)
    p.add_argument("--model", choices=models.NAMES, default=config.MODEL)
    p.add_argument("--chips", action="store_true", help="also play chips, using the config.py thresholds")
    p.add_argument("--no-plan", action="store_true",
                   help="hold one squad for the whole horizon instead of planning transfers week by week")
    p.add_argument("--price-weight", type=float, default=config.PRICE_WEIGHT,
                   help="xP per GBP million of expected price change (0 ignores price rises)")
    p.add_argument("--pool-size", type=int, default=config.POOL_SIZE,
                   help="candidates per position given to the optimiser (lower = faster)")
    p.add_argument("--sweep", nargs="+", metavar="KEY=V1,V2",
                   help="run one backtest per combination, e.g. --sweep horizon=3,5,8 discount=0.8,0.9")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("markets", help="fetch betting-market odds (Polymarket) for every match since 2024-25")
    p.add_argument("--refresh", action="store_true", help="re-download price histories already cached")
    p.set_defaults(func=cmd_markets)

    p = sub.add_parser("archive", help="copy the downloaded data into archive/ (fetch and markets do this as they go)")
    p.add_argument("--deadline", action="store_true",
                   help="snapshot every player's price, news and ownership for the next deadline instead")
    p.add_argument("--no-markets", action="store_true", help="skip Polymarket (it needs the event listing)")
    p.set_defaults(func=cmd_archive)

    for name, func, help_ in (("export", cmd_export, "write the website's data to web/public/data/"),
                              ("publish", cmd_publish, "export, build the website and push it to GitHub Pages")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--team-id", type=int, help="the FPL team to show (default: the one saved in the dashboard)")
        p.add_argument("--model", choices=models.NAMES, default=config.MODEL,
                       help="model whose xP to show where no forecast was saved before a deadline")
        p.set_defaults(func=func)
        if name == "publish":
            p.add_argument("--no-export", action="store_true", help="use the data already exported")
            p.add_argument("--build-only", action="store_true", help="build web/dist/ but don't push it")

    p = sub.add_parser("scorecard", help="score this season's saved pre-deadline forecasts against results")
    p.add_argument("--season", help="default: the current season")
    p.set_defaults(func=cmd_scorecard)

    p = sub.add_parser("app", help="open the interactive dashboard in your browser")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    p.set_defaults(func=cmd_app)

    for name, func, help_ in (("predict", cmd_predict, "xP for upcoming gameweeks"),
                              ("recommend", cmd_recommend, "team, transfers, captain and chip advice")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--horizon", type=int, default=config.HORIZON)
        p.add_argument("--model", choices=models.NAMES, default=config.MODEL)
        p.set_defaults(func=func)
        if name == "predict":
            p.add_argument("--top", type=int, default=10)
        else:
            p.add_argument("--team-id", type=int, help="your FPL team id (from the Points page URL)")
            p.add_argument("--free-transfers", type=int, help="override the estimated free transfers")
            p.add_argument("--bank", type=float, help="override money in the bank (£m)")
            p.add_argument("--max-hits", type=int, default=config.MAX_HITS,
                           help=f"max extra transfers at -{config.HIT_COST} each")
            p.add_argument("--no-plan", action="store_true",
                           help="don't plan the following weeks' transfers, just this week's")
            p.add_argument("--active-chip", choices=["wildcard", "freehit"],
                           help="a chip you've already activated for the upcoming GW (not visible via the API)")
            p.add_argument("--budget", type=float, default=100.0, help="budget when building from scratch")
            p.add_argument("--sensitivity", type=int, default=0, metavar="N",
                           help="re-solve N times with noisy xP and report how often each move wins")

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
