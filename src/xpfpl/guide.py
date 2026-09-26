"""The dashboard's Guide tab: how xP-FPL works and what the terms mean, for FPL players new to ML."""

import altair as alt
import pandas as pd
import streamlit as st

from xpfpl import backtest, config, models, tune, validate
from xpfpl.features import FEATURES, PLAYER_STATS, SEQ_LEN, WINDOWS
from xpfpl.style import POINTS

# Two series, so two hues: the model and the yardstick it has to beat. Grey is for reference
# marks only (perfect hindsight, the y = x line), never for a series.
MLP_COLOUR = "#2a78d6"
BASE_COLOUR = "#b8541a"
MUTED = "#8a8984"
MODEL_LABELS = {
    "mlp": "MLP",
    "components": "Components",
    "embed": "Embedding",
    "sequence": "Sequence",
    "gbm": "LightGBM",
    "xmins": "xMins",
    "ensemble": "Ensemble",
    "baseline": "Baseline",
    "FPL xP": "FPL's own xP",
}


def _names(report: dict) -> tuple[str, str]:
    """The model this report is about, and the yardstick it is compared with."""
    return report.get("primary", "MLP"), report.get("reference", "Baseline")


def _scale(report: dict) -> alt.Scale:
    primary, reference = _names(report)
    return alt.Scale(domain=[primary, reference], range=[MLP_COLOUR, BASE_COLOUR])

def _midweek_factors() -> str:
    """The default model's midweek factors in words, if `xpfpl train` / `xpfpl cups` has fitted them."""
    from xpfpl.data import cups
    table = cups.factors(config.MODEL)
    if not table:
        return "They haven't been fitted yet: run `xpfpl cups` or retrain."
    parts = [f"{cups.ROTATION_LABELS.get(k, k)} ×{v:.2f}" for k, v in table.items()]
    return f"For the default model ({config.MODEL}): " + "; ".join(parts) + "."


GLOSSARY = [
    ("xP (expected points)",
     "The model's prediction of how many FPL points a player will score in a gameweek. It's an "
     "average over many possible outcomes, so a 5.0 xP player might blank or haul. Double gameweeks "
     "add both fixtures; blank gameweeks are 0."),
    ("MLP (model: mlp)",
     "Multi-layer perceptron, the simplest kind of neural network. It takes "
     f"{len(FEATURES)} numbers about a player and fixture and passes them through two layers of learned "
     "weights (128 then 64 units) to produce one number: expected points. It learns those weights "
     "from every player-match since 2016-17."),
    ("Baseline (model: baseline)",
     "Each player's average points over their last 5 matches, with no machine learning. It is the "
     "yardstick: any model is only worth using because it beats this."),
    ("Component model (model: components)",
     "Instead of predicting one number directly, it predicts scoring components separately: whether "
     "a player will appear, reach 60 minutes, score goals, provide assists, make saves or earn bonus "
     "points. It then combines those estimates using FPL's scoring rules, making the expected-points "
     "forecast easier to interpret."),
    ("Embedding model (model: embed)",
    "The MLP plus a learned vector for every player and club (`nn.Embedding`). The player vector lets "
    "the model represent persistent player-specific tendencies beyond recent form; the club vector "
    "does the same for club-specific context such as playing style and typical team strength. New "
    "signings and promoted clubs share a catch-all vector until they have enough matches."),
    ("Sequence model (model: sequence)",
     f"A GRU (a small recurrent network) that reads the last {SEQ_LEN} matches in order rather than "
     "averaging them, so 2-2-12 and 5-4-3 look different to it even though both average 5."),
    ("LightGBM (model: gbm)",
     "Gradient-boosted decision trees on the same features - the standard tool for table-shaped "
        "data, included as an additional model for comparison with the neural-network models."),
    ("xMins model (model: xmins)",
     "Predicts minutes first - no minutes, a cameo (1-59) or 60+ - and then the points each of those "
     "is worth, so its xP is P(cameo) x cameo points + P(60+) x full-game points. It also gives each "
     "player's expected minutes and chance of playing, shown in the Players table."),
    ("Ensemble (model: ensemble)",
     "The average of the MLP, LightGBM and xMins predictions. Different models make different "
     "mistakes, so the average is usually as good as the best of them and steadier."),
    ("Midweek factor",
     "When a club plays a cup or European match before a gameweek, each of its players' xP for that "
     "gameweek is multiplied by a factor for his own role in it, once the match has been played. "
     "Regular starters (60+ minutes a match lately) barely move, rested or not; squad players who "
     "weren't used score less than the model expects that weekend, and those who played score more. "
     "Fitted on 2025-26 onwards, where the cup and European data starts, and only for the next "
     "gameweek: later weeks show the midweek matches but aren't adjusted. " + _midweek_factors()),
    ("FPL's own xP",
     "The expected-points figure the official game shows for each player, taken as it stood before "
     "each match. It is essentially recent form, so it is a yardstick, not a rival model."),
    ("Features",
     "The inputs the model sees. Here: averages of a player's recent stats (last 3, 5 and 10 "
     "matches) and per-90 rates over the last 20 and 38; points at the same venue; team and opponent "
     "goals and xG; the fixture's expected goals from fitted team attack/defence ratings; the crowd's "
     "net transfers and ownership; where the player ranks at his club by price and minutes; and home "
     "or away, price, position and experience."),
    ("Training / validation",
     "Training is when the model adjusts its weights to fit past matches. Validation checks it on a "
     "season it never saw (2025-26), which is the honest test of how it will do on new games. Once "
     "that's checked, the saved model is refitted on all seasons so it knows the latest form."),
    ("Epoch / early stopping",
     "One epoch is one pass through all the training data. Early stopping ends training when the "
     "validation score stops improving, so the model doesn't memorise the past (overfitting)."),
    ("RMSE / MAE",
     "Accuracy scores in points. MAE (mean absolute error) is the typical miss: MAE 1.8 means "
     "predictions are about 1.8 points off on average. RMSE (root mean squared error) punishes big "
     "misses more than MAE. Lower is better for both."),
    ("Calibration",
     "Whether the numbers mean what they say: of all the players the model called 3 xP, did they "
     "average three points? A model can rank players well and still be badly calibrated, which "
     "would make every xP gain in the app look bigger than it is."),
    ("Captain test",
     "A football-sized way of scoring the model: each gameweek, captain whoever the model rates "
     "highest in the league and count what they scored. Compared with perfect hindsight and with a "
     "typical starting player, so you can see the edge in points rather than in decimals."),
    ("Backtest",
     "Replaying a past season deadline by deadline, using only the data that existed at the time: "
    "picking a squad, making transfers, applying transfer penalties and scoring each week for real. It is the only "
     "test that measures the whole app rather than the model alone, and what the horizon, discount "
     "and chip thresholds get tuned with (`xpfpl tune`)."),
    ("Tuning",
    "`xpfpl tune` sets the tuning parameters in config.py by backtesting candidate values over several "
    "seasons and keeping whatever scores most points, one group of parameters at a time. It's the "
     "difference between 'a 5-gameweek horizon feels right' and 'a 5-gameweek horizon scored best'."),
    ("In-sample",
     "Predicting matches the model was trained on. Past-gameweek xP in the review tab is in-sample, "
     "so it looks a little more accurate than the model really is on future games."),
    ("Injury scaling",
     "xP is multiplied by FPL's chance-of-playing flag (a 50% flag halves xP). For later weeks, "
     "flagged players are assumed to recover by 25 percentage points per gameweek."),
    ("Optimiser",
     "After predicting xP, an integer linear program (the PuLP library with the CBC solver) searches "
     "every legal squad for the highest total xP: £ budget, 2 GKP / 5 DEF / 5 MID / 3 FWD, max 3 per "
    "club, valid formations, free transfers and transfer penalties. It picks the transfers, XI, bench and captain."),
    ("Horizon",
     f"How many gameweeks ahead the optimiser plans for (default {config.HORIZON}). Longer horizons avoid "
     "chasing one good fixture; shorter ones react faster to form."),
    ("Planning transfers",
     "With week-by-week planning on, the optimiser chooses a squad for every gameweek in the "
    "horizon, banking free transfers and applying transfer penalties exactly as the game does. What you get "
     "is a route - 'roll this week, take Haaland next' - of which only this week's move is acted "
     "on; the rest is re-planned next week from fresh predictions."),
    ("Price change / team value",
     "Players rise or fall £0.1m as managers buy and sell them, and form is a decent predictor of "
     f"which way. With a price weight above zero (now {config.PRICE_WEIGHT:g}) the optimiser treats £1m "
     "of expected value as worth that many points, breaking ties towards a player about to rise."),
    ("Discount",
     f"Later weeks count for less because predictions further out are less certain. Each week ahead is "
     f"worth {config.DISCOUNT} times the one before, so GW+2 counts {config.DISCOUNT ** 2:.2f} as much as this week."),
    ("Bench weight",
     f"Bench points only count through auto-subs, so bench xP is valued at {config.BENCH_WEIGHT:.0%}. "
     "That's why the optimiser prefers cheap benches unless Bench Boost is active."),
    ("Free transfer value",
     f"Rolling an unused free transfer is treated as worth {config.FT_VALUE} xP, so the optimiser only "
     "makes a transfer if it gains more than that."),
    ("Transfer penalty",
     f"Each transfer beyond your free ones costs {config.HIT_COST} points. 'Max transfer penalties' caps how many the "
     "optimiser may take."),
    ("xP gain vs no transfers",
     "Expected points of the recommended plan minus keeping your squad unchanged, over the horizon, "
    "after transfer penalties."),
    ("Chip gain / threshold",
     "Each chip gets a score: extra xP from playing it this week. It's recommended only if that beats "
     f"a threshold (Triple Captain {config.TRIPLE_CAPTAIN_MIN_XP:g}, Bench Boost {config.BENCH_BOOST_MIN_XP:g}, "
     f"Free Hit +{config.FREE_HIT_MIN_GAIN:g}, Wildcard +{config.WILDCARD_MIN_GAIN:g}) and no later week in the "
     "same chip window looks better. The thresholds come from `xpfpl tune`."),
    ("Selling price",
     "What you'd get for a player: purchase price plus half of any rise (rounded down to £0.1m). "
     "Falls are passed on in full. The optimiser budgets with these, not current prices."),
    ("Hindsight best XI",
     "In the review tab: the best XI and captain you could have picked from the same 15 players if "
     "you'd known the actual points. The gap is the points left on the table from selection alone."),
    ("FDR (fixture difficulty)",
     "FPL's own 1 (easy) to 5 (hard) rating per fixture, shown in the fixture ticker. The model "
     "doesn't use it; it uses team goals and xG, and each fixture's expected goals from fitted "
     "attack/defence ratings, instead."),
    ("xP per £m",
     "Total xP over the horizon divided by price. It's a value measure for finding cheap enablers."),
]


COLUMN_HELP = [
    ("MAE", "Mean absolute error: how far a typical prediction is from the points scored.",
    "Lower is better. This is the clearest measure of the model's typical prediction error."),
    ("RMSE", "Root mean squared error: the same miss, but big misses are punished far more.",
    "Lower is better. A larger gap above MAE indicates that unusually large errors have a greater "
    "effect on the result."),
    ("All players", "Every player in every matchday squad, including the ones who never came on.",
    "This includes many players with little or no match involvement. It is useful for comparing "
    "models on the same complete set of player-match records, but can make accuracy appear stronger."),
    ("Players getting minutes", "Only players who appeared in at least one of their previous five matches.",
    "This is the more relevant measure for player selection because it focuses on players with a "
    "recent record of playing. Lower error indicates more reliable expected-points estimates."),
    ("R²", "The share of the week-to-week variation in points that the predictions explain.",
    "Higher is better, but expect small numbers: even a simulated perfect model only reaches about "
    "0.2 for players getting minutes. Most of a week's points are luck."),
    ("Rank corr", "Spearman correlation between predicted and actual points within each gameweek, averaged.",
    "Higher is better. The optimiser only needs the order of the players to be right, so this is "
    "the measure closest to what drives the team selection."),
    ("Bias", "The average of (prediction - points scored) across the rows.",
    "Near zero is preferable. A positive bias means the model tends to overestimate points; a "
    "negative bias means it tends to underestimate them."),
]


def _metric_row(head: pd.DataFrame, report: dict) -> None:
    primary, reference = _names(report)
    active = head[head["subset"] == "Players getting minutes"].set_index("model")
    close = next((e["share"] for e in report["errors"] if e["band"] == "within 1"), float("nan"))
    cap = pd.DataFrame(report["captain"])
    m = st.columns(4)
    m[0].metric("Typical miss (MAE)", f"{active.at[primary, 'mae']:.2f} pts",
                f"{active.at[primary, 'mae'] - active.at[reference, 'mae']:+.2f} vs baseline",
                delta_color="inverse", help="Players getting minutes. Lower is better.")
    m[1].metric("Big-miss score (RMSE)", f"{active.at[primary, 'rmse']:.2f} pts",
                f"{active.at[primary, 'rmse'] - active.at[reference, 'rmse']:+.2f} vs baseline",
                delta_color="inverse", help="Hauls and blanks weigh heaviest here. Lower is better.")
    m[2].metric("Within 1 point", f"{close:.0%}",
                help="Share of predictions that landed within a point of the real score.")
    if len(cap):
        m[3].metric("Captain pick", f"{cap[primary].mean():.2f} pts/GW",
                    f"{cap[primary].mean() - cap['typical'].mean():+.2f} vs a typical starter",
                    help="Points scored by the highest-xP player in the league each gameweek.")


def _calibration_chart(rows: list[dict]) -> alt.Chart:
    """Predicted vs actual: the dashed line is a perfect model, points below it are optimism."""
    df = pd.DataFrame(rows)
    df["low"] = df["actual"] - 1.96 * df["se"]
    df["high"] = df["actual"] + 1.96 * df["se"]
    hi = float(max(df["predicted"].max(), df["high"].max())) + 0.4
    axis = alt.Scale(domain=[0, hi], nice=False)
    perfect = alt.Chart(pd.DataFrame({"v": [0.0, hi]})).mark_line(
        color=MUTED, strokeDash=[4, 4], strokeWidth=2).encode(x="v:Q", y="v:Q")
    ci = alt.Chart(df).mark_rule(color=MLP_COLOUR, strokeWidth=2, opacity=0.35).encode(
        x=alt.X("predicted:Q", scale=axis), y=alt.Y("low:Q", scale=axis), y2="high:Q")
    line = alt.Chart(df).mark_line(color=MLP_COLOUR, strokeWidth=2, point=alt.OverlayMarkDef(
        color=MLP_COLOUR, size=64, filled=True)).encode(
        x=alt.X("predicted:Q", title="What the model predicted (xP)", scale=axis),
        y=alt.Y("actual:Q", title="What they really scored", scale=axis),
        tooltip=[alt.Tooltip("bin:N", title="xP band"), alt.Tooltip("predicted:Q", format=".2f"),
                 alt.Tooltip("actual:Q", title="actual", format=".2f"),
                 alt.Tooltip("n:Q", title="players", format=",")])
    return (perfect + ci + line).properties(height=260)


def _decile_chart(rows: list[dict], report: dict) -> alt.Chart:
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("decile:O", title="Predicted xP: lowest tenth of players to highest",
                axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("model:N", sort=list(_names(report)),
                            scale=alt.Scale(paddingInner=0.2)),  # a gap between the paired bars
        y=alt.Y("actual:Q", title="Points they really scored (average)"),
        color=alt.Color("model:N", scale=_scale(report), title=None,
                        legend=alt.Legend(orient="top-left", direction="horizontal")),
        tooltip=[alt.Tooltip("model:N"), alt.Tooltip("decile:O", title="tenth"),
                 alt.Tooltip("predicted:Q", title="predicted", format=".2f"),
                 alt.Tooltip("actual:Q", title="actual", format=".2f")]).properties(height=260)


def _gameweek_chart(rows: list[dict], report: dict) -> alt.Chart:
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40, filled=True)).encode(
        x=alt.X("gw:O", title="Gameweek", axis=alt.Axis(values=list(range(1, 39, 4)))),
        y=alt.Y("mae:Q", title="Typical miss that week (MAE)", scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", scale=_scale(report), title=None,
                        legend=alt.Legend(orient="top", direction="horizontal")),
        tooltip=[alt.Tooltip("model:N"), alt.Tooltip("gw:O", title="GW"),
                 alt.Tooltip("mae:Q", format=".2f"), alt.Tooltip("n:Q", title="players")]
    ).properties(height=260)


def _captain_chart(rows: list[dict], report: dict) -> alt.Chart:
    """One measure, four ways of choosing a captain: a single-series bar, labelled directly."""
    primary, reference = _names(report)
    cap = pd.DataFrame(rows)
    df = pd.DataFrame({
        "pick": ["Perfect hindsight", "The model's top pick", "The baseline's top pick",
                 "A typical starting player"],
        "points": [cap["best"].mean(), cap[primary].mean(), cap[reference].mean(), cap["typical"].mean()],
    })
    base = alt.Chart(df).encode(
        y=alt.Y("pick:N", title=None, sort=list(df["pick"]), axis=alt.Axis(labelLimit=200)),
        x=alt.X("points:Q", title="Average points scored by the captain, per gameweek"))
    bars = base.mark_bar(color=MLP_COLOUR, cornerRadiusTopRight=3, cornerRadiusBottomRight=3, height=22).encode(
        tooltip=[alt.Tooltip("pick:N", title=None), alt.Tooltip("points:Q", format=".2f")])
    labels = base.mark_text(align="left", dx=6, color="#4a4944").encode(text=alt.Text("points:Q", format=".2f"))
    return (bars + labels).properties(height=170)


def _error_chart(rows: list[dict]) -> alt.Chart:
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_bar(color=MLP_COLOUR, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("band:N", title="Points scored minus points predicted", sort=list(df["band"]),
                axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("share:Q", title="Share of predictions", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("band:N", title="miss"), alt.Tooltip("share:Q", format=".1%"),
                 alt.Tooltip("count:Q", title="players", format=",")]).properties(height=220)


def render_accuracy() -> None:
    st.subheader("How accurate are the models?")
    report = validate.load_report()
    if not report:
        st.info("No accuracy report yet. Click **Retrain** in the sidebar (or run `xpfpl validate`) "
                "to score the model on a season it has never seen.")
        return

    st.markdown(
        f"The models were trained on **{report['trained_on']}** and then evaluated on every match of "
        f"**{report['season']}**, a season not used for training. The results compare their predictions "
        "with what really happened and include a no-machine-learning baseline based on each player's "
        "average over their last five matches.")

    head = pd.DataFrame(report["headline"])
    _metric_row(head, report)

    comparison = _same_season_comparison(report)
    everyone = head
    if comparison and comparison.get("headline"):
        others = pd.DataFrame(comparison["headline"])
        others["model"] = others["model"].replace({"baseline": validate.REFERENCE})   # same model, two labels
        everyone = pd.concat([head, others[~others["model"].isin(head["model"])]], ignore_index=True)
    metrics = [m for m in ("mae", "rmse", "r2", "spearman") if m in everyone]
    table = everyone.pivot(index="model", columns="subset", values=metrics)
    table.columns = [f"{METRIC_LABELS.get(m, m.upper())}, {s.upper()}" for m, s in table.columns]
    accuracy_table = table.reset_index().rename(columns={"model": "MODEL"})
    sort_by = "RMSE, PLAYERS GETTING MINUTES"
    if sort_by in accuracy_table:
        accuracy_table = accuracy_table.sort_values(sort_by)
    accuracy_table["MODEL"] = accuracy_table["MODEL"].map(lambda name: MODEL_LABELS.get(name, name))
    st.dataframe(accuracy_table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in table.columns})
    primary, reference = _names(report)
    if len(everyone["model"].unique()) > len(head["model"].unique()):
        st.caption(f"Every model from the latest `xpfpl compare`, all scored on {report['season']}. The "
                   f"charts below compare the default model ({MODEL_LABELS.get(primary, primary)}) with "
                   "the baseline.")
    else:
        st.caption("Run `xpfpl compare` to add every other model to this table.")
    with st.expander("What these columns mean"):
        st.table(pd.DataFrame(COLUMN_HELP, columns=["COLUMN", "WHAT IT MEASURES", "HOW TO READ IT"]))
        st.markdown("Both metrics are expressed in **FPL points per player per match**, and lower is better. "
                    "The 'players getting minutes' columns are the most relevant when assessing player-selection "
                    "performance.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Does 6 xP really mean six points?**")
        st.altair_chart(_calibration_chart(report["calibration"]), width="stretch")
        st.caption("Players grouped by predicted points. If the blue line follows the dashed line, "
               "predicted points are well calibrated. A lower line at higher predictions indicates "
               "that the model tends to overestimate the highest forecasts. The bars show the margin "
               "of error.")
    with c2:
        st.markdown("**Does it rank players effectively?**")
        st.altair_chart(_decile_chart(_pair(report["deciles"], report), report), width="stretch")
        st.caption("Players are split into ten groups by predicted points, from the lowest-ranked group to "
               "the highest-ranked group. Higher actual scores in the higher-ranked groups indicate that "
               "the model is ordering players effectively. The comparison with the baseline shows whether "
               "the model adds value beyond recent average points.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Is it steady all season, or only in the good weeks?**")
        st.altair_chart(_gameweek_chart(_pair(report["by_gameweek"], report), report), width="stretch")
        st.caption("The typical miss, gameweek by gameweek. A model that only works in quiet weeks "
                   "would spike in the chaotic ones. This stays in a narrow band and beats the baseline "
                   "in most weeks, including blank and double gameweeks.")
    with c2:
        st.markdown("**The captain test**")
        st.altair_chart(_captain_chart(report["captain"], report), width="stretch")
        st.caption("Each gameweek, captain the highest-xP player in the league and see what they scored. "
                   "No model gets near perfect hindsight, and nothing ever will: that is the noise in "
                   "football, not a flaw in the maths. What matters is the gap over a typical starter.")

    st.markdown("**How often is it close?**")
    st.altair_chart(_error_chart(report["errors"]), width="stretch")
    st.caption("Points scored minus points predicted for players getting minutes. The central bars represent "
         "the most common prediction errors, while the right tail represents larger underestimates. "
         "When projected points are close, consider team news and other information not available "
         "to the model.")

    _render_extras(report, comparison)

    pos = pd.DataFrame(report["by_position"])
    with st.expander("Performance by position"):
        position_table = pos.pivot(index="position", columns="model", values=["mae", "rmse", "bias"])
        position_table.columns = [f"{metric.upper()}, {MODEL_LABELS.get(model, model)}"
                      for metric, model in position_table.columns]
        st.dataframe(position_table.reset_index().rename(columns={"position": "POSITION"}),
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in position_table.columns})
        st.caption(f"Report generated {report['generated'][:16].replace('T', ' ')} from "
                   f"{report['active_rows']:,} player-matches. Rebuild it with **Retrain** or "
                   "`xpfpl validate`. Bias is prediction minus actual: positive means optimistic.")


METRIC_LABELS = {"mae": "MAE", "rmse": "RMSE", "r2": "R²", "spearman": "RANK CORR"}


def _same_season_comparison(report: dict) -> dict | None:
    """The latest `xpfpl compare`, if it scored the same held-out season as this report."""
    comparison = validate.load_comparison()
    return comparison if comparison and comparison.get("season") == report.get("season") else None


def _pair(rows: list[dict], report: dict) -> list[dict]:
    """Only the default model and the baseline: the charts compare those two."""
    keep = set(_names(report))
    return [r for r in rows if r.get("model") in keep]


def _render_extras(report: dict, comparison: dict | None = None) -> None:
    """The ceiling, forecasts made further ahead, and errors by what actually happened.

    Uses the latest `xpfpl compare` for the multi-model views when it covers the same season."""
    ceiling = report.get("ceiling") or (comparison or {}).get("ceiling")
    if ceiling:
        st.markdown("**How good could any model be?**")
        st.markdown(
            f"Simulating every match {ceiling['sims']} times from the component model's probabilities, a "
            f"*perfect* model (one that knew the true chances) would still score RMSE "
            f"**{ceiling['rmse_median']:.2f}** ({ceiling['rmse_p5']:.2f} to {ceiling['rmse_p95']:.2f}) and R² "
            f"**{ceiling['r2_median']:.3f}** on these players. The gap between that and the table above is "
            "all the room left to improve; the rest is luck.")
    horizons = pd.DataFrame((comparison or {}).get("horizons") or report.get("horizons") or [])
    if len(horizons):
        st.markdown("**How far ahead can it see?**")
        horizons["model"] = horizons["model"].map(lambda name: MODEL_LABELS.get(name, name))
        chart = alt.Chart(horizons).mark_line(point=True).encode(
            x=alt.X("horizon:O", title="Gameweeks ahead the forecast was made"),
            y=alt.Y("rmse:Q", title="RMSE (players getting minutes)", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title=None),
            tooltip=["model", "horizon", alt.Tooltip("rmse:Q", format=".3f"),
                     alt.Tooltip("r2:Q", format=".3f")]).properties(height=220)
        st.altair_chart(chart, width="stretch")
        st.caption("The same matches, predicted from what was known 1, 2 and 3 gameweeks before. "
                   "Forecasts fade quickly, which is why the optimiser discounts later weeks.")
    groups = pd.DataFrame((comparison or {}).get("return_groups") or report.get("return_groups") or [])
    if len(groups):
        with st.expander("Errors by what actually happened"):
            groups["model"] = groups["model"].map(lambda name: MODEL_LABELS.get(name, name))
            group_table = groups.pivot(index="group", columns="model", values="rmse")
            st.dataframe(group_table.reset_index(), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.3f")
                                        for c in group_table.columns})
            st.caption("RMSE over every row, split by what each player actually scored. Misses on players who "
                       "didn't play are about minutes; misses on hauls are about spotting the big scores.")


def render_backtest() -> None:
    st.subheader("What would it have scored?")
    runs = backtest.load_results()
    if not runs:
        st.info("No season replay yet. Run `xpfpl backtest --season 2024-25` in the terminal (a few "
                "minutes) to replay a whole season gameweek by gameweek, picking the squad with only "
                "the data available at each deadline.")
        return

    st.markdown(
        "Accuracy scores are one thing; points on the board are another. A backtest replays a past "
        "season from the first deadline to the last, each week rebuilding form from matches already "
        "played, predicting, transferring and picking a captain, then scoring the week for real, "
        "auto-subs and transfer penalties included.")

    labels = {f"{r['settings']['season']} · {r['settings']['model']} · horizon "
              f"{r['settings']['horizon']} · discount {r['settings']['discount']:g}"
              f"{' · chips' if r['settings']['chips'] else ''}": r for r in runs}
    choice = st.selectbox("Replay", list(labels))
    run, gws = labels[choice], pd.DataFrame(labels[choice]["gameweeks"])
    s = run["summary"]

    m = st.columns(5)
    m[0].metric("Season total", f"{s['points']:.0f} pts")
    m[1].metric("Per gameweek", f"{s['points_per_gw']:.2f}")
    m[2].metric("Transfers", f"{s['transfers']}",
                f"-{s['hit_cost']} pts in penalties" if s["hits"] else "no penalties",
                delta_color="inverse" if s["hits"] else "off")
    m[3].metric("From the armband", f"{s['captain_points']:.0f} pts",
                help="Captain's own score. It counts twice in the total (three times on a Triple Captain).")
    m[4].metric("Left on the bench", f"{s['bench_points']:.0f} pts")

    gws["running"] = gws["points"].cumsum()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Points per gameweek**")
        base = alt.Chart(gws).encode(x=alt.X("gw:O", title="Gameweek",
                                             axis=alt.Axis(values=list(range(1, 39, 4)))))
        bars = base.mark_bar(color=MLP_COLOUR, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            y=alt.Y("points:Q", title="Points (after penalties)"),
            tooltip=[alt.Tooltip("gw:O", title="GW"), alt.Tooltip("points:Q", format=".0f"),
                     alt.Tooltip("xp:Q", title="predicted", format=".2f"),
                     alt.Tooltip("captain_name:N", title="captain"),
                     alt.Tooltip("transfers:Q"), alt.Tooltip("chip:N")])
        mean = alt.Chart(pd.DataFrame({"v": [gws["points"].mean()]})).mark_rule(
            color=MUTED, strokeDash=[4, 4], strokeWidth=2).encode(
                y=alt.Y("v:Q", title="Season average"))
        st.altair_chart(bars + mean, width="stretch")
    with c2:
        st.markdown("**Running total**")
        line = alt.Chart(gws).mark_area(
            color=MLP_COLOUR, opacity=0.15, line={"color": MLP_COLOUR, "strokeWidth": 2}).encode(
            x=alt.X("gw:O", title="Gameweek", axis=alt.Axis(values=list(range(1, 39, 4)))),
            y=alt.Y("running:Q", title="Points so far"),
            tooltip=[alt.Tooltip("gw:O", title="GW"), alt.Tooltip("running:Q", title="total", format=".0f")])
        st.altair_chart(line, width="stretch")

    st.caption(f"Predicted {s['xp']:.0f} points across the season and scored {s['points'] + s['hit_cost']:.0f} "
               f"before transfer penalties ({s['xp_error_per_gw']:+.2f} per gameweek). Historical replays do not include "
               "the injury and availability information available before a current deadline, so the total "
               "should be interpreted as a conservative estimate. Comparisons between settings are the most "
               "informative use of the result.")

    if len(runs) > 1:
        st.markdown("**All replays**")
        # Replays saved by older versions may not carry every column, so reindex rather than index.
        st.dataframe(pd.DataFrame([r["summary"] for r in runs]).reindex(columns=[
            "season", "model", "horizon", "discount", "ft_value", "max_hits", "plan_transfers",
            "chips", "gws", "points", "points_per_gw", "hits", "transfers"]).rename(columns={
                "season": "Season", "model": "Model", "horizon": "Horizon", "discount": "Discount",
                "ft_value": "FT value", "max_hits": "Max transfer penalties", "plan_transfers": "Plans ahead",
                "chips": "Chips", "gws": "GWs",
                "points": "Points", "points_per_gw": "Per GW", "hits": "Penalties",
                "transfers": "Transfers"}).sort_values("Points", ascending=False),
            hide_index=True, width="stretch",
            column_config={"Per GW": st.column_config.NumberColumn(format=POINTS),
                           "Points": st.column_config.NumberColumn(format="%.0f")})
        st.caption("Compare like with like: the same season and model, one tuning parameter at a time. "
                   "`xpfpl backtest --sweep horizon=3,5,8` runs a batch and ranks them, and "
                   "`xpfpl tune` searches them all and picks the winners.")

    st.dataframe(gws[["gw", "points", "gross", "hits", "transfers", "chip", "xp", "captain_name",
                      "captain_points", "bench_points", "autosubs", "squad_value", "bank"]].rename(columns={
        "gw": "GW", "points": "Net", "gross": "Scored", "hits": "Penalties", "transfers": "Transfers",
        "chip": "Chip", "xp": "Predicted", "captain_name": "Captain", "captain_points": "Capt pts",
        "bench_points": "Bench", "autosubs": "Auto-subs", "squad_value": "Value £m", "bank": "Bank £m"}),
        hide_index=True, width="stretch", height=280,
        column_config={"Predicted": st.column_config.NumberColumn(format=POINTS),
                       "Value £m": st.column_config.NumberColumn(format="%.1f"),
                       "Bank £m": st.column_config.NumberColumn(format="%.1f")})


def render_models() -> None:
    """Every model that has been compared on the same held-out season."""
    report = validate.load_comparison()
    if not report:
        return
    st.subheader("Which model?")
    table = pd.DataFrame(report["models"])
    table["description"] = table["model"].map(models.DESCRIPTIONS).fillna(table["description"])
    table["model"] = table["model"].map(lambda name: MODEL_LABELS.get(name, name))
    st.markdown(
        f"Each model was trained on the same seasons and asked to predict **{report['season']}**, which "
        f"none of the models had seen ({report['rows']:,} player-matches from players getting minutes). "
        "The model used by default is set by `config.MODEL`; `xpfpl compare` rebuilds this table.")
    columns = {"model": "MODEL", "description": "WHAT IT IS", "rmse": "RMSE", "mae": "MAE", "r2": "R²",
               "spearman": "Rank corr", "captain_pts_per_gw": "Captain pts/GW", "epochs": "Epochs",
               "seconds": "Train (s)"}
    shown = table[[c for c in columns if c in table]].rename(columns=columns)
    st.dataframe(shown.sort_values("RMSE"), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in ("RMSE", "MAE", "R²", "Rank corr") if c in shown})
    st.caption("Differences of a few hundredths of a point are noise. The captain column is the more "
               "football-shaped test: the points scored by each model's top pick each week.")


def render_tuning() -> None:
    """What `xpfpl tune` chose, how much each setting was worth, and whether it held up on unseen seasons."""
    from xpfpl import robustness

    report = tune.load_report()
    if not report:
        st.info("The tuning parameters in config.py haven't been tuned on this machine. `xpfpl tune` replays "
                "whole seasons with candidate values and prints the config block to paste in.")
        return
    st.subheader("How the tuning parameters were chosen")
    trials = pd.DataFrame(report["trials"])
    st.markdown(
        f"The settings in config.py (how many weeks to look ahead, what a banked free transfer is worth, the "
        f"chip thresholds...) were tried out by replaying **{', '.join(report['seasons'])}** with the "
        f"`{report['model']}` model, a group of related settings at a time ({len(trials)} candidate seasons).")

    rows = []
    for stage, group in trials.groupby("stage", sort=False):
        best = group.loc[group["mean_points"].idxmax()]
        parameters = [c for c in group.columns if c in tune.CONFIG_NAMES and group[c].notna().any()]
        rows.append({"Stage": stage,
                     "Best in that search": ", ".join(f"{k}={best[k]}" for k in parameters),
                     "Points a season": best["mean_points"],
                     "Worth (best - worst)": group["mean_points"].max() - group["mean_points"].min()})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.0f")
                                for c in ("Points a season", "Worth (best - worst)")})
    st.caption("Read the last column with care: one replay of a season moves by about 84 points on luck alone "
               "(see \"Does it hold up?\" below), so a setting worth less than that in a single-replay search "
               "is a guess, not a finding.")

    retune = robustness.latest_retune()
    if retune:
        table = pd.DataFrame([{"Settings": c["label"], "Points a season": c["mean_points"], "±": c["se"],
                               **{s: p for s, p in (c["points_per_season"] or {}).items()}}
                              for c in retune["confirmation"]])
        st.markdown(
            f"**Checked on seasons the search never saw.** A later, more careful search ({retune['replays']} "
            f"replays per candidate, `{retune['model']}` model, tuned on {', '.join(retune['seasons'])}) was then "
            f"replayed on **{', '.join(retune['confirm_seasons'])}** against today's settings:")
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.0f")
                                    for c in table.columns if c != "Settings"})
        st.caption("\"tuned\" is what that search picked, \"current config\" is what the app uses, \"pre-tuning\" "
                   "the original guesses. They finish within noise of each other, so config.py was left as it is: "
                   "these settings matter far less than the forecasts. The ones that clearly did matter: valuing "
                   "expected price rises, a near-zero bench weight, playing the chips, and not using the Free Hit "
                   "on a small gain.")


def render_robustness() -> None:
    """Six seasons, each predicted by a model trained only on the seasons before it (`xpfpl robustness`)."""
    from xpfpl import robustness

    s = robustness.site_summary(robustness.load_report())
    if not s:
        return
    st.subheader("Does it hold up?")
    seasons = s["seasons"]
    st.markdown(
        f"The accuracy above is one season. `xpfpl robustness` repeats the test on **{len(seasons)} seasons** "
        f"({seasons[0]} to {seasons[-1]}): each is forecast by a model trained only on the seasons before it, "
        "then replayed week by week.")

    acc = pd.DataFrame(s["accuracy"])
    acc = acc[acc["model"].isin(["ensemble", "gbm", "mlp", "baseline", "fpl_xp"])]
    acc["model"] = acc["model"].map(lambda m: MODEL_LABELS.get(m, {"fpl_xp": "FPL's own xP"}.get(m, m)))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Forecast error each season** (RMSE, lower is better)")
        st.altair_chart(alt.Chart(acc).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("season:O", title=None), y=alt.Y("rmse:Q", title="RMSE", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title=None, legend=alt.Legend(orient="bottom")),
            tooltip=["season", "model", alt.Tooltip("rmse:Q", format=".3f"), alt.Tooltip("spearman:Q", format=".3f")]),
            width="stretch")
    bt = s["backtest"]
    points = pd.DataFrame(bt["points"]).T if bt["points"] else pd.DataFrame()
    with c2:
        if len(points):
            st.markdown("**Points replaying each season**")
            long = points.reset_index(names="variant").melt(id_vars="variant", var_name="season", value_name="points")
            long = long[long["variant"].isin(["ensemble", "baseline", "oracle"])].replace(
                {"variant": {"ensemble": "The model", "baseline": "5-match average", "oracle": "Perfect foresight"}})
            st.altair_chart(alt.Chart(long.dropna()).mark_line(point=True, strokeWidth=2).encode(
                x=alt.X("season:O", title=None), y=alt.Y("points:Q", title="Points", scale=alt.Scale(zero=False)),
                color=alt.Color("variant:N", title=None, legend=alt.Legend(orient="bottom")),
                tooltip=["season", "variant", alt.Tooltip("points:Q", format=".0f")]), width="stretch")

    lines = []
    ens = next((r for r in s["spread"] if r["model"] == "ensemble"), None)
    if ens:
        lines.append(f"**Steady from season to season:** RMSE {ens['rmse_mean']:.3f} ± {ens['rmse_sd']:.3f} "
                     f"({ens['rmse_min']:.3f} to {ens['rmse_max']:.3f}).")
    if s["calibration"]:
        cal = s["calibration"]
        lines.append(f"**xP means what it says:** over all seasons, points = {cal['intercept']:+.2f} + "
                     f"{cal['slope']:.2f} × xP (a perfect forecast would be 0 + 1.00 × xP).")
    if s["p_play"]:
        lines.append(f"**The chance of playing is well judged:** Brier score {s['p_play']['brier']:.3f}, against "
                     f"{s['p_play']['brier_naive']:.3f} for \"share of his last five matches played\".")
    if s["seeds"]:
        same = pd.DataFrame(s["seeds"])["captain_same"].mean()
        lines.append(f"**The captain pick is often a coin toss:** retrained from different random starting points, "
                     f"the model's top captain is the same in {same:.0%} of weeks. The top options are that close.")
    gaps = {g["variant"]: g for g in bt["gaps"]}
    if "baseline" in gaps and bt.get("noise_sd"):
        b = gaps["baseline"]
        lines.append(f"**Worth having:** the model beat the 5-match average by {-b['gap_mean']:.0f} points a season "
                     f"in the replays, in {b['seasons_worse']} of {b['seasons']} seasons. But one replay moves by "
                     f"about {bt['noise_sd']:.0f} points a season on luck alone, so smaller gaps (between models, or "
                     "between settings) can't be told apart.")
    if s["leak_probe"] and all(p["leaking"] == 0 for p in s["leak_probe"]):
        lines.append("**No peeking at the future:** rebuilding every feature from only the matches before a deadline "
                     "gives the same numbers as the full build.")
    st.markdown("\n".join(f"- {line}" for line in lines))
    if len(points):
        with st.expander("Every replay variant"):
            table = points.copy()
            table["Mean"] = table.mean(axis=1)
            st.dataframe(table.sort_values("Mean", ascending=False), width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.0f") for c in table.columns})
            st.caption("Each season's points for the same model under different rules: 'oracle' knows every result "
                       "(the ceiling), 'pre-tuning settings' uses the original guesses, 'chips on' plays chips too.")


def render() -> None:
    st.header("How xP-FPL works")
    st.markdown(
        "Every gameweek it predicts points for every player, then searches for the best legal squad, "
        "XI and captain. Four steps:")
    c = st.columns(4)
    steps = [
        ("1. Data", "Historical player and fixture information comes from the vaastav archive, while the "
                "current season is updated from the FPL API. **Fetch match data** updates the dataset."),
        ("2. Features", f"For each player and fixture, the system combines recent performance statistics "
                f"averaged over the last {', '.join(map(str, WINDOWS))} matches with team and opponent "
                "strength, home or away status, price and position."),
        ("3. Predict", f"A model turns those numbers into expected points (xP) per fixture - by default "
                       f"`{config.MODEL}`, one of {len(models.NAMES)} you can train and compare. **Retrain** "
                       "refits it on the latest data."),
        ("4. Decide", "An optimiser picks transfers, XI, bench, captain and chips to maximise xP over the "
                      "next few gameweeks, within FPL's rules. By default it holds one squad across those "
                      "weeks: planning transfers week by week is an option, but it scored fewer points "
                      "when past seasons were replayed."),
    ]
    for slot, (title, body) in zip(c, steps):
        with slot.container(border=True):
            st.markdown(f"**{title}**  \n{body}")

    st.subheader("Weekly routine")
    st.markdown(
        "1. Once the sidebar says the gameweek is **complete** (FPL has confirmed its data, usually a few "
        "hours after the last match): **Fetch match data**, **Fetch betting odds**, then **Retrain** "
        "(a few minutes).\n"
        "2. **Gameweek Review**: see how the week went and where the model was wrong.\n"
        "3. Before the deadline: **Refresh live FPL data** for the latest prices and injury flags, then "
        "check **Players & Fixtures** and **Markets** for team news.\n"
        "4. **Plan Ahead**: if you've activated a chip, set it first. Add injury or rotation worries to "
        "**Never pick**, then click **Update plan**.\n"
        "5. Make the moves on the FPL site. The app can't make transfers for you.\n\n"
        "**Fetch match data** and **Refresh live FPL data** both do two things. "
        "Fetch rebuilds the saved training dataset from finished matches, so the model needs retraining "
        "afterwards. Refresh only re-downloads what changes day to day (prices, injury flags, your team) "
        "for the pages you're looking at. It saves nothing and needs no retraining.")

    render_accuracy()
    render_models()
    render_backtest()
    render_robustness()
    render_tuning()

    st.subheader("Glossary")
    search = st.text_input("Filter terms", placeholder="e.g. mlp, horizon, penalty")
    for term, text in GLOSSARY:
        if not search or search.lower() in (term + text).lower():
            with st.expander(term):
                st.write(text)

    st.subheader("Known limitations")
    st.markdown(
        "- Chips you've activated and transfers you've made aren't visible via the FPL API until the "
        "deadline, so set the active chip yourself.\n"
        "- Later weeks of the plan are a route, not a commitment: they assume today's predictions and "
        "get re-planned every week.\n"
        "- Free transfers are estimated by replaying your season. Override them if they look wrong.\n"
        "- The model doesn't read team news, press conferences or predicted line-ups: only FPL's injury flag.\n"
        "- Price changes are predicted from form and transfer momentum rather than the precise criteria "
        "used by FPL, so treat them as a tie-breaker.\n"
        "- Blank and double gameweek handling still hasn't been tested against a real one.")
