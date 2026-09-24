"""Replay a past season one gameweek at a time, using only what was known at each deadline.

This is the honest end-to-end test: not "how close is xP to the points scored" (that's
validate.py) but "how many points would this app have scored me". Every gameweek it

  1. rebuilds each player's form from matches played *before* that deadline,
  2. predicts xP for the next `horizon` gameweeks from that snapshot,
  3. runs the same optimiser the app runs, with the free transfers and bank it has accrued,
    4. plays the gameweek: applies auto-subs, counts the captain twice and applies transfer penalties.

Because every tuning parameter (horizon, discount, transfer-penalty limit, chip thresholds) changes the answer in
points rather than in RMSE, this is what you tune them with: `xpfpl backtest --sweep`.

Simplifications, all of which make it a little optimistic or a little pessimistic but not
unfair: no injury flags (they aren't in the historical data, so a flagged player looks
available), prices move with the real data but the squad's value doesn't earn from price
rises the manager didn't make, the pool is the players who appear in a matchday squad that
season, and chip windows follow the current two-halves rule.
"""

import json
from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from xpfpl import chips, config, models, prices, teams as team_ratings
from xpfpl.data.history import load_matches
from xpfpl.features import (FEATURES, PLAYER_STATE, TARGET, TEAM_STATE, add_market_features, add_row_features,
                            build_training_frame, fixture_strength)
from xpfpl.models.trainer import TrainConfig
from xpfpl.myteam import selling_price
from xpfpl.optimise import shortlist, solve

PLAYER_STATE_COLS = PLAYER_STATE
_RATINGS: dict = {}


def _ratings_before(season: str, gw: int) -> pd.DataFrame:
    """Team ratings as fitted just before `season` GW `gw` (one row per club)."""
    if "table" not in _RATINGS:
        _RATINGS["table"] = team_ratings.ratings(load_matches())
    table = _RATINGS["table"]
    known = table[(table["season"] == season) & (table["gw"] <= gw)]
    if known.empty:        # before the season's first fit: the latest earlier ratings, if any
        return table[table["season"] < season].groupby("team_code").tail(1).set_index("team_code")
    return known[known["gw"] == known["gw"].max()].set_index("team_code")


CHIP_WINDOWS = ((1, 19), (20, 38))            # two of each chip, as in 2025-26 onwards
CHIP_ORDER = ("3xc", "bboost", "freehit", "wildcard")


@dataclass
class Settings:
    """One backtest configuration. The defaults mirror config.py, i.e. what the app does today."""
    season: str = "2024-25"
    start_gw: int = 1
    end_gw: int = 38
    horizon: int = config.HORIZON
    discount: float = config.DISCOUNT
    bench_weight: float = config.BENCH_WEIGHT
    ft_value: float = config.FT_VALUE
    max_hits: int = config.MAX_HITS
    budget: float = 100.0
    model: str = config.MODEL
    chips: bool = False
    plan_transfers: bool = config.PLAN_TRANSFERS   # plan a squad per gameweek across the horizon
    price_weight: float = config.PRICE_WEIGHT      # xP per £m of expected price change
    pool_size: int = config.POOL_SIZE              # candidates kept per position, to keep the ILP quick
    thresholds: dict = field(default_factory=lambda: {
        "3xc": config.TRIPLE_CAPTAIN_MIN_XP, "bboost": config.BENCH_BOOST_MIN_XP,
        "freehit": config.FREE_HIT_MIN_GAIN, "wildcard": config.WILDCARD_MIN_GAIN})

    def label(self) -> str:
        parts = [self.season, self.model, f"h{self.horizon}", f"d{self.discount:g}",
                 f"ft{self.ft_value:g}", f"bw{self.bench_weight:g}", f"hits{self.max_hits}"]
        if self.plan_transfers:
            parts.append("plan")
        if self.price_weight:
            parts.append(f"pw{self.price_weight:g}")
        parts.append("chips" if self.chips else "nochips")
        return "_".join(parts)


# ---------------------------------------------------------------- the model, as of a season

def fit_model(frame: pd.DataFrame, season: str, epochs: int = 40, name: str = config.MODEL,
              quiet: bool = False):
    """Train on every season before `season`, exactly as the real pipeline would have then.

    The season before that one is the early-stopping holdout; the model is then refit on all
    of the earlier data for the epoch count that worked best. Any model in xpfpl.models works.
    """
    if name == "baseline":
        return None
    history = frame[frame["season"].str[:4].astype(int) < int(season[:4])]
    if history.empty:
        raise ValueError(f"No seasons before {season} to train on.")
    holdout = history["season"].max()
    earlier = history["season"].str[:4].astype(int) < int(holdout[:4])
    fit = models.fit(name, history[earlier], history[history["season"] == holdout],
                     cfg=TrainConfig(epochs=epochs), quiet=quiet)
    if not quiet:
        print(f"Refitting on {len(history):,} rows up to {holdout} "
              f"for {fit.meta['best_epoch']} epochs...")
    return models.fit(name, history, None, cfg=models.refit_config(fit), quiet=quiet)


# ---------------------------------------------------------------- what was known at a deadline

def _snapshot(rows: pd.DataFrame, gw: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each player's and each club's form as of the deadline of `gw`.

    A row's own rolling features already exclude that match, so the last row up to and
    including `gw` is the right pre-deadline state. A player blank in `gw` keeps the state
    from their previous fixture, which is one match stale but never sees the future.
    """
    known = rows[rows["gw"] <= gw].sort_values("kickoff_time")
    players = known.groupby("element").tail(1).set_index("element")
    teams = known.groupby("team_code").tail(1).set_index("team_code")[[f"team_{c}" for c in TEAM_STATE]]
    return players, teams


def _predict_horizon(rows: pd.DataFrame, state: pd.DataFrame, teams: pd.DataFrame, gws: list[int],
                     predictor, price_table: dict | None = None) -> pd.DataFrame:
    """xP per player per gameweek in `gws`, predicted from the `gw`-deadline snapshot."""
    up = rows[rows["gw"].isin(gws)].copy()
    snap = state.reindex(up["element"])
    # Form as of the deadline, not as of the match. Assigned in one go: column by column would
    # fragment the frame, and there are well over a hundred of them.
    known = [c for c in PLAYER_STATE_COLS if c in state.columns]
    replacements = {c: snap[c].to_numpy() for c in known}
    replacements["price"] = snap["value"].to_numpy() / 10.0
    for prefix, code in (("team", "team_code"), ("opp", "opp_code")):
        form = teams.reindex(up[code])
        for c in TEAM_STATE:
            replacements[f"{prefix}_{c}"] = np.nan_to_num(form[f"team_{c}"].to_numpy(), nan=1.35)
    # Fixture strength from the ratings known at this deadline, not at each later match.
    strength = fixture_strength(up, _ratings_before(up["season"].iloc[0], min(gws)))
    replacements.update({c: strength[c].to_numpy() for c in strength.columns})
    up[list(replacements)] = pd.DataFrame(replacements, index=up.index)
    # Odds exist only for this deadline's gameweek: later weeks take the fallback, as they would live.
    later = up["gw"] > min(gws)
    if later.any():
        up = pd.concat([up[~later], add_market_features(up[later], None, use=False)]).loc[up.index]
    up = add_row_features(up)                           # venue/role features follow the new state
    up[FEATURES] = up[FEATURES].fillna(0.0).astype("float32")
    up = up.copy()                                      # defragment after the bulk assignments

    xp = models.load("baseline").predict(up) if predictor is None else predictor.predict(up)
    up = up.assign(xp=np.clip(xp.astype(float), 0.0, None))
    wide = up.pivot_table(index="element", columns="gw", values="xp", aggfunc="sum")  # doubles add up
    wide = wide.reindex(index=state.index, columns=gws).fillna(0.0)                   # blanks are 0
    wide.columns = [f"xp_{g}" for g in gws]

    players = pd.DataFrame({
        "name": state["name"], "team": state["team"], "position": state["position"].astype(int),
        "price": state["value"] / 10.0, "now_cost": state["value"].astype(int),
    }).join(wide)
    players["xp_total"] = players[[f"xp_{g}" for g in gws]].sum(axis=1)
    if price_table is not None:
        # Expected price drift, from a table fitted only on the seasons before this one.
        players["total_points_r3"] = state.get("total_points_r3", pd.Series(0.0, index=state.index))
        players["price_delta"] = prices.expected_delta(players, price_table)
    return players


def _shortlist(players: pd.DataFrame, squad: dict, per_position: int) -> pd.DataFrame:
    """The optimiser's shortlist, plus whoever we already own (who must stay priceable)."""
    return shortlist(players, list(squad), per_position)


# ---------------------------------------------------------------- playing the gameweek out

def apply_autosubs(xi: list[int], bench: list[int], position: dict[int, int],
                   minutes: dict[int, float]) -> tuple[list[int], list[tuple[int, int]]]:
    """FPL's auto-subs: bench players replace starters who didn't play, in bench order."""
    xi, subs = list(xi), []
    for sub in bench:
        if minutes.get(sub, 0) <= 0:
            continue
        for out in [p for p in xi if minutes.get(p, 0) <= 0]:
            if position[sub] == 1 or position[out] == 1:
                if position[sub] != position[out]:      # the goalkeeper only swaps with a goalkeeper
                    continue
            else:
                counts = Counter(position[p] for p in xi if p != out)
                counts[position[sub]] += 1
                if not all(config.LINEUP_MIN[p] <= counts[p] <= config.LINEUP_MAX[p]
                           for p in config.SQUAD_SIZE):
                    continue
            xi[xi.index(out)] = sub
            subs.append((out, sub))
            break
    return xi, subs


def score_gameweek(plan, gw: int, points: dict[int, float], minutes: dict[int, float],
                   position: dict[int, int], chip: str | None) -> dict:
    """Actual points for a plan's team: auto-subs, captaincy (vice if the captain blanked), chips."""
    xi, subs = apply_autosubs(plan.lineups[gw], plan.bench[gw], position, minutes)
    captain = plan.captains[gw]
    if minutes.get(captain, 0) <= 0 and minutes.get(plan.vice_captains[gw], 0) > 0:
        captain = plan.vice_captains[gw]
    if captain not in xi:                                # both blanked: no captain points
        captain = None
    multiplier = 3 if chip == "3xc" else 2
    bench = [p for p in plan.squad if p not in xi]
    scored = sum(points.get(p, 0.0) for p in xi)
    scored += (multiplier - 1) * points.get(captain, 0.0) if captain else 0.0
    if chip == "bboost":
        scored += sum(points.get(p, 0.0) for p in bench)
    return {"points": scored, "captain": captain, "captain_points": points.get(captain, 0.0) if captain else 0.0,
            "bench_points": sum(points.get(p, 0.0) for p in bench), "autosubs": len(subs)}


# ---------------------------------------------------------------- the season loop

def _available_chips(used: dict[str, list[int]], gw: int) -> dict[str, dict]:
    out = {}
    for start, stop in CHIP_WINDOWS:
        if not start <= gw <= stop:
            continue
        for name in CHIP_ORDER:
            if not any(start <= e <= stop for e in used.get(name, [])):
                out[name] = {"name": name, "start_event": start, "stop_event": stop}
    return out


def _pick_chip(players, gws, plan, available, solve_kwargs, thresholds):
    """The app's chip rule, with the thresholds under test. Returns (chip name or None, advice)."""
    advice = chips.advise(players, gws, plan, available, solve_kwargs)
    for a in advice:
        a.threshold = thresholds.get(a.chip, a.threshold)
    best = next((a for a in advice if a.gain >= a.threshold and not a.better_later), None)
    return (best.chip if best else None), advice


def run(settings: Settings, frame: pd.DataFrame | None = None, predictor=None,
        verbose: bool = True, price_table: dict | None = None) -> "Result":
    """Replay `settings.season` and return the per-gameweek record plus a summary."""
    frame = build_training_frame(load_matches()) if frame is None else frame
    rows = frame[frame["season"] == settings.season]
    if rows.empty:
        raise ValueError(f"No data for {settings.season}. Seasons: {sorted(frame['season'].unique())}")
    if settings.model == "baseline":
        predictor = None
    elif predictor is None:
        predictor = fit_model(frame, settings.season, name=settings.model)
    if settings.price_weight and price_table is None:
        price_table = prices.fit(frame, before_season=settings.season)

    actual = rows.groupby(["gw", "element"])[[TARGET, "minutes"]].sum()
    position = rows.groupby("element")["position"].first().astype(int).to_dict()
    season_gws = sorted(rows["gw"].unique())
    last = min(settings.end_gw, max(season_gws))

    squad: dict[int, int] = {}          # element -> purchase price in tenths
    bank, free_transfers, used_chips = settings.budget, 1, {}
    records = []

    for gw in [g for g in season_gws if settings.start_gw <= g <= last]:
        gws = [g for g in season_gws if gw <= g < gw + settings.horizon]
        state, teams = _snapshot(rows, gw)
        pool = _predict_horizon(rows, state, teams, gws, predictor, price_table)
        now_cost = pool["now_cost"].to_dict()   # not `prices`: that name is the module
        selling = {p: selling_price(now_cost.get(p, buy), buy) for p, buy in squad.items()}
        pool = _shortlist(pool, squad, settings.pool_size)

        plan_settings = dict(discount=settings.discount, bench_weight=settings.bench_weight,
                     ft_value=settings.ft_value, price_weight=settings.price_weight, time_limit=60)
        base = dict(current_squad=selling or None, bank=round(bank, 1), **plan_settings)
        plan = solve(pool, gws, **base, free_transfers=free_transfers, max_hits=settings.max_hits,
                     plan_transfers=settings.plan_transfers)

        chip, advice = (None, [])
        if settings.chips and squad:
            chip, advice = _pick_chip(pool, gws, plan, _available_chips(used_chips, gw),
                                      {k: v for k, v in base.items()}, settings.thresholds)
        if chip == "3xc":
            plan = solve(pool, gws, **base, free_transfers=free_transfers, max_hits=settings.max_hits,
                         plan_transfers=settings.plan_transfers, triple_captain_gw=gw)
        elif chip == "bboost":
            plan = solve(pool, gws, **base, free_transfers=free_transfers, max_hits=settings.max_hits,
                         plan_transfers=settings.plan_transfers, bench_boost_gw=gw)
        elif chip == "freehit":
            plan = solve(pool, [gw], **base, unlimited_transfers=True)
        elif chip == "wildcard":
            plan = solve(pool, gws, **base, unlimited_transfers=True)
        if chip:
            used_chips.setdefault(chip, []).append(gw)

        had_squad = bool(squad)
        gw_actual = actual.loc[gw] if gw in actual.index.get_level_values(0) else actual.iloc[:0]
        points = gw_actual[TARGET].to_dict()
        minutes = gw_actual["minutes"].to_dict()
        result = score_gameweek(plan, gw, points, minutes, position, chip)

        hits = 0 if chip in ("wildcard", "freehit") else plan.hits
        moves = len(plan.transfers_in) if squad else 0
        net = result["points"] - config.HIT_COST * hits
        records.append({
            "gw": gw, "points": net, "gross": result["points"], "hits": hits, "transfers": moves,
            "chip": chip or "", "xp": plan.xp[gw], "captain": plan.captains[gw],
            "captain_name": pool.at[result["captain"], "name"] if result["captain"] else "-",
            "captain_points": result["captain_points"], "bench_points": result["bench_points"],
            "autosubs": result["autosubs"], "free_transfers": free_transfers,
            "bank": round(bank, 1), "squad_value": round(sum(now_cost.get(p, v) for p, v in squad.items()) / 10.0, 1)
            if squad else settings.budget,
        })
        if verbose:
            print(f"  GW{gw:<3d} xP {plan.xp[gw]:5.1f}  actual {result['points']:5.1f}  "
                  f"hits -{config.HIT_COST * hits:<2d} net {net:5.1f}  transfers {moves}  "
                  f"C: {records[-1]['captain_name']} ({result['captain_points']:.0f})"
                  f"{'  ' + chip.upper() if chip else ''}")

        # Settle up: a Free Hit squad is handed back at the end of the week.
        if chip != "freehit":
            for p in plan.transfers_out:
                bank += selling[p]
            for p in plan.transfers_in:
                bank -= now_cost[p] / 10.0
                squad[p] = now_cost[p]
            for p in plan.transfers_out:
                squad.pop(p, None)
            if not squad:
                squad = {p: now_cost[p] for p in plan.squad}
                bank = settings.budget - sum(squad.values()) / 10.0
        if had_squad:  # building the opening squad is free and doesn't earn next week's transfer
            used = 0 if chip in ("wildcard", "freehit") else moves
            free_transfers = min(max(free_transfers - used, 0) + 1, config.MAX_FREE_TRANSFERS)

    return Result(settings, pd.DataFrame(records))


@dataclass
class Result:
    settings: Settings
    gameweeks: pd.DataFrame

    @property
    def summary(self) -> dict:
        g = self.gameweeks
        return {
            **{k: v for k, v in asdict(self.settings).items() if k != "thresholds"},
            "gws": int(len(g)),
            "points": float(g["points"].sum()),
            "points_per_gw": float(g["points"].mean()),
            "hits": int(g["hits"].sum()),
            "hit_cost": int(config.HIT_COST * g["hits"].sum()),
            "transfers": int(g["transfers"].sum()),
            "xp": float(g["xp"].sum()),
            "xp_error_per_gw": float((g["gross"] - g["xp"]).mean()),
            "captain_points": float(g["captain_points"].sum()),
            "bench_points": float(g["bench_points"].sum()),
            "chips": ", ".join(f"{r.chip} GW{r.gw}" for r in g[g["chip"] != ""].itertuples()),
        }

    def save(self) -> "tuple[object, object]":
        config.BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        stem = self.settings.label()
        csv = config.BACKTEST_DIR / f"{stem}.csv"
        js = config.BACKTEST_DIR / f"{stem}.json"
        self.gameweeks.to_csv(csv, index=False, encoding="utf-8")
        js.write_text(json.dumps({"summary": self.summary, "settings": asdict(self.settings),
                                  "gameweeks": self.gameweeks.to_dict("records")}, indent=1), encoding="utf-8")
        return csv, js


def load_results(directory=None) -> list[dict]:
    """Every saved backtest, newest first, for the dashboard to show without re-running anything.

    The directory also holds the tuning report, which is not a replay, so anything without a
    summary and a list of gameweeks is skipped.
    """
    directory = directory or config.BACKTEST_DIR
    out = []
    for path in sorted(directory.glob("*.json"), key=lambda p: -p.stat().st_mtime) if directory.exists() else []:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(result, dict) and {"summary", "settings", "gameweeks"} <= set(result):
            out.append(result)
    return out


def predictors(frame: pd.DataFrame, season: str, names, quiet: bool = False) -> dict:
    """One trained model per name, so a sweep or a tuning run trains each of them only once."""
    return {name: fit_model(frame, season, name=name, quiet=quiet) for name in dict.fromkeys(names)}


def sweep(base: Settings, grid: dict[str, list], frame: pd.DataFrame | None = None,
          verbose: bool = False, save: bool = True) -> pd.DataFrame:
    """Run one backtest per combination in `grid` (e.g. {"horizon": [3, 5, 8]}) and rank them."""
    from itertools import product

    frame = build_training_frame(load_matches()) if frame is None else frame
    seasons = grid.get("season", [base.season])
    fitted = {s: predictors(frame, s, grid.get("model", [base.model])) for s in seasons}
    tables = {s: prices.fit(frame, before_season=s) for s in seasons}
    rows = []
    for combo in product(*grid.values()):
        settings = Settings(**{**asdict(base), **dict(zip(grid, combo))})
        print(f"\n{settings.label()}")
        result = run(settings, frame=frame, predictor=fitted[settings.season][settings.model],
                     price_table=tables[settings.season], verbose=verbose)
        if save:
            result.save()
        rows.append(result.summary)
        print(f"  -> {rows[-1]['points']:.0f} points ({rows[-1]['points_per_gw']:.1f}/GW, "
              f"{rows[-1]['hits']} hits)")
    return pd.DataFrame(rows).sort_values("points", ascending=False)
