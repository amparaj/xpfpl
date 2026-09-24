"""The dashboard's Markets tab: what the betting markets (Polymarket) think, next to the model.

"Market Odds" are Polymarket's prices; "Our Odds" are the model's own team ratings (teams.py).
For a gameweek that has been played, every view also shows what actually happened, so the two
can be judged. The only market numbers the model itself uses are the team-level ones, through
features.py, and the "ruled out" signal from the scorer markets, in predict.py.
"""

from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from xpfpl import config, teams
from xpfpl.data import api, markets
from xpfpl.data.history import load_matches
from xpfpl.features import _poisson_win
from xpfpl.style import DECIMAL, PERCENT, club_columns

HOME, DRAW, AWAY = "#2a78d6", "#b9b8b3", "#d6552a"     # stacked win/draw/loss bars
OURS, MARKET, ACTUAL = "#8a8984", "#2a78d6", "#111111"
SOURCES = ["Market Odds", "Our Odds", "Actual"]
OUTRIGHT_ORDER = ["Champion", "Top 4", "Top 5", "Top 6", "Relegated", "Last Place", "Top Goalscorer",
                  "Most Assists", "Most Clean Sheets", "Player of the Season", "Sacked"]
UPCOMING = "Next matches (live odds)"


# ---------------------------------------------------------------- cached data

@st.cache_data(ttl=600, show_spinner="Fetching live odds...")
def _upcoming(bs: dict, fixtures: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return markets.upcoming(bs, fixtures)


@st.cache_data(ttl=600, show_spinner="Fetching price history...")
def _history(slug: str, days: float, end: pd.Timestamp) -> pd.DataFrame:
    return markets.match_history(slug, days, end)


@st.cache_data(show_spinner=False)
def _accuracy(stamp: float) -> pd.DataFrame:
    market, _ = markets.load()
    return markets.accuracy(market, load_matches()) if market is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _ratings(stamp: float, season: str, gw: int) -> pd.DataFrame:
    """Our ratings as they stood before `gw` (for a played gameweek) or today (for the next one)."""
    table = teams.ratings(load_matches())
    known = table[(table["season"] == season) & (table["gw"] == gw)]
    return known.set_index("team_code") if len(known) else teams.latest(load_matches(), season, gw)


@st.cache_data(show_spinner=False)
def _results(stamp: float, season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per fixture: goals and xG for each side. Per player per fixture: goals and minutes."""
    m = load_matches()
    m = m[m["season"] == season]
    side = (m.groupby(["fixture", "team_code"])
             .agg(xg=("expected_goals", lambda s: s.sum(min_count=1)), was_home=("was_home", "first"),
                  home_goals=("team_h_score", "first"), away_goals=("team_a_score", "first")).reset_index())
    side["goals"] = np.where(side["was_home"], side["home_goals"], side["away_goals"])
    home = side[side["was_home"]].rename(columns={"team_code": "home_code", "xg": "xg_home", "goals": "goals_home"})
    away = side[~side["was_home"]].rename(columns={"team_code": "away_code", "xg": "xg_away", "goals": "goals_away"})
    fixtures = home[["fixture", "home_code", "xg_home", "goals_home"]].merge(
        away[["fixture", "away_code", "xg_away", "goals_away"]], on="fixture")
    players = m.groupby(["fixture", "code"], as_index=False)[["goals_scored", "minutes"]].sum()
    return fixtures, players


@st.cache_data(ttl=3600, show_spinner="Pricing the season markets at the start of the season...")
def _season_start(tokens: tuple[str, ...], when: pd.Timestamp) -> dict[str, float]:
    return markets.outright_prices_at(list(tokens), when)


LOCAL_TZ = datetime.now().astimezone().tzinfo


def _local(ts) -> pd.Timestamp:
    """A timestamp in this computer's time zone, as the rest of the app shows times."""
    return pd.Timestamp(ts).tz_convert(LOCAL_TZ)


def _stamp() -> float:
    return config.MATCHES_PATH.stat().st_mtime if config.MATCHES_PATH.exists() else 0.0


# ---------------------------------------------------------------- charts

def _win_bars(table: pd.DataFrame) -> alt.Chart:
    long = table.melt(id_vars=["label", "order"], value_vars=["home_win", "draw", "away_win"],
                      var_name="result", value_name="probability")
    long["result"] = long["result"].map({"home_win": "Home Team win", "draw": "Draw", "away_win": "Away Team win"})
    return alt.Chart(long).mark_bar().encode(
        y=alt.Y("label:N", title=None, sort=table.sort_values("order")["label"].tolist(),
                axis=alt.Axis(labelLimit=260)),
        x=alt.X("probability:Q", stack="normalize", title="Market Odds for each result", axis=alt.Axis(format="%")),
        color=alt.Color("result:N", title=None, scale=alt.Scale(domain=["Home Team win", "Draw", "Away Team win"],
                                                                range=[HOME, DRAW, AWAY]),
                        legend=alt.Legend(orient="top")),
        order=alt.Order("order_result:Q"),
        tooltip=[alt.Tooltip("label:N", title="match"), "result", alt.Tooltip("probability:Q", format=".0%")],
    ).transform_calculate(order_result="datum.result == 'Home Team win' ? 0 : datum.result == 'Draw' ? 1 : 2"
    ).properties(height=alt.Step(28))


def _dumbbell(rows: pd.DataFrame, cols: dict[str, str], title: str, fmt: str) -> alt.Chart:
    """One row per club: Market Odds and Our Odds joined by a line, plus Actual where known."""
    present = {k: v for k, v in cols.items() if k in rows and rows[k].notna().any()}
    long = rows.melt(id_vars=["club", "vs", "venue"], value_vars=list(present), var_name="source", value_name="value")
    long["source"] = long["source"].map(present)
    first = next(iter(present))
    order = rows.sort_values(first, ascending=False)["club"].tolist()
    odds = long[long["source"] != "Actual"]
    line = alt.Chart(odds).mark_rule(color="#cfcdc7", strokeWidth=2).encode(
        y=alt.Y("club:N", sort=order, title=None), x="min(value):Q", x2="max(value):Q")
    dots = alt.Chart(long).mark_point(filled=True, size=90, opacity=1).encode(
        y=alt.Y("club:N", sort=order, title=None),
        x=alt.X("value:Q", title=title, axis=alt.Axis(format=fmt)),
        color=alt.Color("source:N", title=None, scale=alt.Scale(domain=SOURCES, range=[MARKET, OURS, ACTUAL]),
                        legend=alt.Legend(orient="top")),
        shape=alt.Shape("source:N", scale=alt.Scale(domain=SOURCES, range=["circle", "circle", "diamond"]),
                        legend=None),
        tooltip=["club", "vs", "venue", "source", alt.Tooltip("value:Q", format=fmt)])
    return (line + dots).properties(height=alt.Step(20))


# ---------------------------------------------------------------- the tab

def _gameweeks(saved: pd.DataFrame | None, season: str, live: pd.DataFrame) -> list:
    options = [UPCOMING] if len(live) else []
    if saved is not None and len(saved):
        played = saved[(saved["season"] == season) & (saved["kickoff"] < pd.Timestamp.now(tz="UTC"))]
        options += [int(g) for g in sorted(played["gw"].unique(), reverse=True)]
    return options


def _with_ours(table: pd.DataFrame, season: str, gw: int) -> pd.DataFrame:
    ratings = _ratings(_stamp(), season, gw)
    gf, ga = teams.fixture_rates(ratings, table["home_code"], table["away_code"], np.ones(len(table)))
    win_home = _poisson_win(gf, ga)
    win_away = _poisson_win(ga, gf)
    return table.assign(ours_home=gf, ours_away=ga, ours_home_win=win_home, ours_away_win=win_away,
                        ours_draw=1 - win_home - win_away)


def _clean_sheet(table: pd.DataFrame, side: str) -> pd.Series:
    """Market Odds of `side` keeping a clean sheet: straight from 'the other side over 0.5' where
    that market exists, else from the fitted expected goals."""
    other = "away" if side == "home" else "home"
    direct = 1 - table.get(f"{other}_over_0.5", pd.Series(np.nan, index=table.index))
    return direct.fillna(np.exp(-table[f"lam_{other}"]))


def render(bs: dict, fixtures: list[dict]) -> None:
    st.subheader("What the betting markets think")
    st.caption("Prediction-market prices from Polymarket: a price of 0.62 means the market "
               "gives it a 62% chance. **Market Odds** are those prices; **Our Odds** are the model's own "
               "team ratings. Match markets usually open about a week before kick-off.")
    season = api.current_season(bs)
    names = {t["code"]: t["short_name"] for t in bs["teams"]}
    saved, saved_scorers = markets.load()
    live, live_scorers = _upcoming(bs, fixtures)

    options = _gameweeks(saved, season, live)
    if not options:
        st.info("No market data yet. Click **Fetch betting odds** in the sidebar (a few minutes the first time).")
        return
    if not len(live):
        st.info("Polymarket hasn't listed the next gameweek's matches yet (they usually appear about a week "
                "before). Pick a played gameweek below to see how the odds compared with what happened.")
    choice = st.selectbox("Gameweek", options, format_func=lambda g: g if g == UPCOMING else f"GW{g} (played)",
                          key="market_gw")
    played = choice != UPCOMING
    if played:
        table = saved[(saved["season"] == season) & (saved["gw"] == choice)].copy()
        scorers = (saved_scorers[saved_scorers["slug"].isin(table["slug"])]
                   if saved_scorers is not None and len(saved_scorers) else pd.DataFrame())
        gw = int(choice)
    else:
        table, scorers = live.copy(), live_scorers
        gw = int(table["gw"].min())

    table = _with_ours(table.sort_values("kickoff").reset_index(drop=True), season, gw)
    table["match"] = table["home_code"].map(names) + " v " + table["away_code"].map(names)
    table["order"] = range(len(table))
    table["cs_home"], table["cs_away"] = _clean_sheet(table, "home"), _clean_sheet(table, "away")
    if played:
        results, player_results = _results(_stamp(), season)
        table = table.merge(results.drop(columns=["home_code", "away_code"]), on="fixture", how="left")
        table["label"] = (table["home_code"].map(names) + " " + table["goals_home"].astype("Int64").astype(str)
                          + "-" + table["goals_away"].astype("Int64").astype(str) + " " + table["away_code"].map(names))
    else:
        player_results = None
        table["label"] = table["match"]

    priced = table["priced_at"].min()
    when = (f"at the FPL deadline, {_local(priced):%a %d %b %H:%M}" if played
            else f"live, {_local(pd.Timestamp.now(tz='UTC')):%a %d %b %H:%M}")
    st.markdown(f"**Win odds** ({when})")
    st.altair_chart(_win_bars(table), width="stretch")
    if played:
        _render_result_check(table)
    else:
        _render_upcoming_table(table)

    _render_team_charts(table, names, played)
    _render_movement(table, played)
    _render_news(table, scorers, names, bs, played, player_results)
    _render_scorers(scorers, bs, played, player_results)
    _render_outrights(bs)
    _render_accuracy()


# ---------------------------------------------------------------- sections

PERCENT_COLUMNS = ("Home Team", "Draw", "Away Team", "CS Home Team", "CS Away Team", "Both score", "Over 2.5")


def _is_percent(column: str) -> bool:
    return column in PERCENT_COLUMNS or column.startswith(
        ("Chance", "Change", "Home Team win", "Draw ", "Away Team win", "Biggest move"))


def _moves_in_points(frame: pd.DataFrame) -> pd.DataFrame:
    """"Change" columns x100: moves are shown signed, in percentage points ("-12.5%")."""
    cols = [c for c in frame.columns if c.startswith("Change") and pd.api.types.is_numeric_dtype(frame[c])]
    return frame.assign(**{c: frame[c] * 100 for c in cols})


def _formats(frame: pd.DataFrame) -> dict:
    out = {}
    for c in frame.columns:
        if c.startswith("Goals"):
            out[c] = st.column_config.NumberColumn(format="%d")
        elif c.startswith("xG"):
            out[c] = st.column_config.NumberColumn(format="%.2f")
        elif c.startswith("Change"):
            out[c] = st.column_config.NumberColumn(format="%+.1f%%")
        elif _is_percent(c) and pd.api.types.is_numeric_dtype(frame[c]):
            out[c] = st.column_config.NumberColumn(format=PERCENT)
        elif c == "Traded $":
            out[c] = st.column_config.NumberColumn(format="compact")
    return out


def _render_upcoming_table(table: pd.DataFrame) -> None:
    shown = table.assign(kickoff=table["kickoff"].map(lambda t: f"{_local(t):%a %d %b %H:%M}"))
    cols = {"match": "Match", "kickoff": "Kick-off", "home_win": "Home Team", "draw": "Draw", "away_win": "Away Team",
            "lam_home": "xG Home Team (Market Odds)", "ours_home": "xG Home Team (Our Odds)",
            "lam_away": "xG Away Team (Market Odds)", "ours_away": "xG Away Team (Our Odds)",
            "cs_home": "CS Home Team", "cs_away": "CS Away Team", "btts": "Both score", "over_2.5": "Over 2.5",
            "volume": "Traded $"}
    shown = shown[[c for c in cols if c in shown]].rename(columns=cols)
    st.dataframe(shown, hide_index=True, width="stretch", column_config=_formats(shown))
    st.caption("xG = expected goals. The Market Odds figure comes from fitting a Poisson model to every goal "
               "market for the match at once (result, total goals, each side's goals, both teams to score). "
               "CS = clean sheet (Market Odds).")


def _render_result_check(table: pd.DataFrame) -> None:
    """Played gameweek: the odds against what happened, match by match and for the week."""
    result = np.select([table["goals_home"] > table["goals_away"], table["goals_home"] == table["goals_away"]],
                       ["home", "draw"], "away")
    codes = pd.Categorical(result, ["home", "draw", "away"]).codes

    def chance(prefix: str) -> np.ndarray:
        return np.choose(codes, [table[f"{prefix}home_win"], table[f"{prefix}draw"], table[f"{prefix}away_win"]])

    p_market, p_ours = chance(""), chance("ours_")
    known = table["xg_home"].notna()
    err_market = (table["lam_home"] - table["xg_home"]).abs() + (table["lam_away"] - table["xg_away"]).abs()
    err_ours = (table["ours_home"] - table["xg_home"]).abs() + (table["ours_away"] - table["xg_away"]).abs()
    closer = lambda a, b: np.where(~known, "-", np.where(a < b, "Market Odds", np.where(b < a, "Our Odds", "Tie")))  # noqa: E731
    results = pd.DataFrame({
        "Match": table["label"],
        "Result": pd.Series(result).map({"home": "Home Team win", "draw": "Draw", "away": "Away Team win"}),
        "Chance of that result (Market Odds)": p_market, "Chance of that result (Our Odds)": p_ours,
        "Outcome": np.where(p_market > p_ours, "Market Odds", np.where(p_ours > p_market, "Our Odds", "Tie")),
    })
    goals = pd.DataFrame({
        "Match": table["label"],
        "xG Home Team (Market Odds)": table["lam_home"], "xG Home Team (Our Odds)": table["ours_home"],
        "xG Home Team (Actual)": table["xg_home"], "Goals Home Team": table["goals_home"],
        "xG Away Team (Market Odds)": table["lam_away"], "xG Away Team (Our Odds)": table["ours_away"],
        "xG Away Team (Actual)": table["xg_away"], "Goals Away Team": table["goals_away"],
        "Outcome": closer(err_market, err_ours),
    })

    def colour(col: pd.Series) -> list[str]:
        if col.name == "Outcome":
            return [f"color: {MARKET}; font-weight: 600" if v == "Market Odds" else
                    "color: #5c5b57; font-weight: 600" if v == "Our Odds" else "" for v in col]
        if col.name.startswith("Chance of that result"):
            return [f"background-color: rgba(42,120,214,{min(v, 1) * 0.45:.2f})" if pd.notna(v) else ""
                    for v in col]
        if col.name.endswith("(Actual)"):
            return ["font-weight: 600"] * len(col)
        return [""] * len(col)

    st.markdown("**The odds against what happened: results**")
    st.dataframe(results.style.apply(colour, axis=0), hide_index=True, width="stretch",
                 column_config=_formats(results))
    st.caption("The chance each gave the result that actually happened.")
    st.markdown("**The odds against what happened: expected goals**")
    st.dataframe(goals.style.apply(colour, axis=0), hide_index=True, width="stretch",
                 column_config=_formats(goals))
    st.caption("xG = expected goals. Market Odds and Our Odds are the forecasts at the deadline; Actual is the xG "
               "each side created in the match, with the goals they scored next to it. 'Outcome' names whichever "
               "forecast missed the actual xG by less, over both sides.")

    n, k = len(table), max(int(known.sum()), 1)
    cs_home, cs_away = (table["goals_away"] == 0).astype(float), (table["goals_home"] == 0).astype(float)
    week = {
        "Result log loss": (float(-np.log(np.clip(p_market, 0.01, 1)).mean()),
                            float(-np.log(np.clip(p_ours, 0.01, 1)).mean())),
        "xG error per side": (float(err_market[known].sum() / (2 * k)), float(err_ours[known].sum() / (2 * k))),
        "Clean-sheet Brier": (float((((table["cs_home"] - cs_home) ** 2).sum()
                                     + ((table["cs_away"] - cs_away) ** 2).sum()) / (2 * n)),
                              float((((np.exp(-table["ours_away"]) - cs_home) ** 2).sum()
                                     + ((np.exp(-table["ours_home"]) - cs_away) ** 2).sum()) / (2 * n))),
    }
    cols = st.columns(len(week))
    for c, (label, (market, ours)) in zip(cols, week.items()):
        c.metric(f"{label} (Market Odds)", f"{market:.3f}", f"{market - ours:+.3f} vs Our Odds ({ours:.3f})",
                 delta_color="inverse")
    st.caption("How the week went; lower is better for all three. Result log loss scores the chance each gave "
               "the actual result; xG error is the average miss against the real xG per side; the clean-sheet "
               "Brier score compares each clean-sheet chance with whether the side kept one. One gameweek is ten "
               "matches, so expect it to swing: the season table at the bottom is the fair test.")


def _render_team_charts(table: pd.DataFrame, names: dict[int, str], played: bool) -> None:
    rows = []
    for us, them in (("home", "away"), ("away", "home")):
        row = pd.DataFrame({
            "club": table[f"{us}_code"].map(names), "vs": table[f"{them}_code"].map(names),
            "venue": "Home Team" if us == "home" else "Away Team",
            "market_cs": table[f"cs_{us}"], "ours_cs": np.exp(-table[f"ours_{them}"]),
            "market_xg": table[f"lam_{us}"], "ours_xg": table[f"ours_{us}"]})
        if played:
            row["actual_cs"] = (table[f"goals_{them}"] == 0).astype(float)
            row["actual_xg"] = table[f"xg_{us}"]
        rows.append(row)
    rows = pd.concat(rows, ignore_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Clean-sheet chances**")
        st.altair_chart(_dumbbell(rows, {"market_cs": "Market Odds", "ours_cs": "Our Odds", "actual_cs": "Actual"},
                                  "Chance of a clean sheet", "%"), width="stretch")
    with c2:
        st.markdown("**Expected goals**")
        st.altair_chart(_dumbbell(rows, {"market_xg": "Market Odds", "ours_xg": "Our Odds", "actual_xg": "Actual"},
                                  "Expected goals scored", ".2f"), width="stretch")
    if played:
        cs = rows["actual_cs"]
        st.caption(f"Actual (black diamond): a clean sheet is 100% if kept and 0% if not ({int(cs.sum())} of "
                   f"{len(cs)} sides kept one); expected goals is the xG the side actually created. Where Market "
                   "Odds and Our Odds disagree, see which the diamond sits nearer.")
    else:
        st.caption("Where the dots disagree, the market knows something our ratings don't (team news, a new "
                   "manager) - or is wrong. The model sees both.")


def _render_movement(table: pd.DataFrame, played: bool) -> None:
    st.markdown("**How the odds moved before the Gameweek deadline**")
    c1, c2 = st.columns([2, 3])
    pick = c1.selectbox("Match", table["label"].tolist(), key="market_match")
    days = c2.radio("Window", markets.MOVE_DAYS, index=3, horizontal=True, key="market_days",
                    format_func=lambda d: f"{_days(d)} before deadline" if played else f"last {_days(d)}")
    row = table[table["label"] == pick].iloc[0]
    deadline = row["priced_at"]
    end = deadline if played else pd.Timestamp.now(tz="UTC").floor("min" if days < 1 else "h")
    history = _history(row["slug"], days, end)
    if not len(history):
        st.caption("No price history for this match yet.")
        return
    lines = alt.Chart(history).mark_line(strokeWidth=2).encode(
        x=alt.X("time:T", title=None), y=alt.Y("probability:Q", title="Market Odds", axis=alt.Axis(format="%")),
        color=alt.Color("outcome:N", title=None, legend=alt.Legend(orient="top")),
        tooltip=["outcome", alt.Tooltip("time:T", format="%a %d %b %H:%M"), alt.Tooltip("probability:Q", format=".1%")])
    chart = lines
    if played:
        marks = pd.DataFrame({"time": [deadline], "what": ["FPL deadline"]})
        rules = alt.Chart(marks).mark_rule(strokeDash=[4, 4], color="#5c5b57").encode(x="time:T", tooltip=["what"])
        text = alt.Chart(marks).mark_text(align="right", dx=-4, y=8, color="#5c5b57").encode(x="time:T", text="what")
        chart = lines + rules + text
    st.altair_chart(chart.properties(height=260), width="stretch")
    every = "Prices every 5 minutes" if days < 1 else "Hourly prices"
    if played:
        st.caption(f"{every} over the {_days(days)} up to the FPL deadline (the dashed line): the prices the "
                   "model and this page use. Anything after the deadline is news a manager can't act on, so it "
                   "isn't shown.")
    else:
        st.caption(f"{every} over the last {_days(days)}, up to now. A sharp move usually means team news.")


def _club_of(scorers: pd.DataFrame, bs: dict) -> pd.Series:
    codes = {t["code"]: t["short_name"] for t in bs["teams"]}
    return scorers["team_code"].map(codes) if "team_code" in scorers else pd.Series("", index=scorers.index)


def _days(n: float) -> str:
    if n < 1:
        return f"{round(n * 24 * 60)} minutes"
    return "1 day" if n == 1 else f"{n} days"


def _render_news(table: pd.DataFrame, scorers: pd.DataFrame, names: dict[int, str], bs: dict, played: bool,
                 player_results: pd.DataFrame | None) -> None:
    """How the odds moved before the prices were taken: how team news shows up."""
    moment = "the Gameweek deadline" if played else "now"
    st.markdown(f"**News in the market odds: how the odds moved before {moment}**")
    days = st.radio("Compared with", markets.MOVE_DAYS, index=2, horizontal=True, key="market_news_days",
                    format_func=lambda d: f"{_days(d)} before the deadline" if played else f"{_days(d)} ago")
    then = f"{_days(days)} before"
    at = "at deadline" if played else "now"
    match_moves, player_moves = markets.movers(table, scorers if scorers is not None and len(scorers) else None, days)
    if len(match_moves):
        m = match_moves.reset_index(drop=True)
        shown = pd.DataFrame({"Home Team": m["home_code"].map(names), "Away Team": m["away_code"].map(names)})
        if played:
            shown["Score"] = (m["goals_home"].astype("Int64").astype(str) + "-"
                              + m["goals_away"].astype("Int64").astype(str))
        for result, label in (("home_win", "Home Team win"), ("draw", "Draw"), ("away_win", "Away Team win")):
            shown[f"{label} {then}"] = m[markets.move_column(result, days)]
            shown[f"{label} {at}"] = m[result]
        shown["Biggest move"] = m["biggest_move"]
        st.caption(f"The chance of each result {_days(days)} before the deadline" if played else
                   f"The chance of each result {_days(days)} ago and now")
        st.dataframe(club_columns(shown, ("Home Team", "Away Team")), hide_index=True, width="stretch",
                     column_config=_formats(shown))
    elif markets.move_column("home_win", days) not in table:
        st.caption(f"No {_days(days)} prices saved yet: click **Fetch betting odds** in the sidebar to add them.")
    else:
        st.caption("No odds history for these matches.")
    if len(player_moves):
        p = player_moves.head(10).reset_index(drop=True)
        shown = _moves_in_points(pd.DataFrame({
            "Player": p["player"], "Club": _club_of(p, bs), f"Chance to score {then}": p["before"],
            f"Chance to score {at}": p["now"], "Change": p["change"]}))
        shown["Status"] = np.where(p["now"] <= markets.OUT_THRESHOLD, "Likely out", "")
        if played and player_results is not None and "code" in p:
            res = p.merge(player_results, on=["fixture", "code"], how="left")
            shown["Played the Match?"] = np.where(res["minutes"].fillna(0).to_numpy() > 0, "Yes", "No")
        st.caption("Goalscorer odds: the players whose chance to score dropped the most. A sharp drop usually "
                   "means an injury or being left out.")
        st.dataframe(club_columns(shown), hide_index=True, width="stretch", column_config=_formats(shown))
    else:
        st.caption("No real moves in the goalscorer odds for these matches.")
    st.caption("Moves are in percentage points. Injuries and rotation reach the market before they reach "
               f"FPL's flags. When a player's scorer odds fall to {markets.OUT_THRESHOLD:.0%} or less he has "
               "almost always been ruled out (on 2025-26, 3 of 62 such players played), so the model cuts his "
               "next-gameweek xP to "
               f"{config.MARKET_OUT_XP_FACTOR:.0%}. A fall from about 50% is usually a new market finding its "
               "price, not news.")


def _render_scorers(scorers: pd.DataFrame, bs: dict, played: bool, player_results: pd.DataFrame | None) -> None:
    st.markdown("**Anytime goalscorer odds**")
    if scorers is None or not len(scorers) or "p_anytime" not in scorers:
        st.caption("No goalscorer markets for these matches.")
        return
    players = pd.DataFrame(bs["elements"]).drop_duplicates("code").set_index("code")[["web_name", "now_cost"]]
    s = scorers.join(players, on="code") if "code" in scorers else scorers.assign(web_name=np.nan, now_cost=np.nan)
    # An untraded market just shows its opening price (usually 50%), which means nothing.
    real = (s["p_anytime"] - 0.5).abs() > 1e-3
    s = s[(s["volume"] > 0) & (real | (s["volume"] >= 1000))].sort_values("p_anytime", ascending=False)
    if s.empty:
        st.caption("No goalscorer market for these matches has traded yet.")
        return
    s = s.reset_index(drop=True)
    shown = pd.DataFrame({"Player": s["web_name"].fillna(s["player"]), "Club": _club_of(s, bs),
                          "Price £m": s["now_cost"] / 10, "Chance to score (Market Odds)": s["p_anytime"],
                          "Traded $": s["volume"]})
    check = played and player_results is not None and "code" in s
    if check:
        res = s.merge(player_results, on=["fixture", "code"], how="left")
        goals, mins = res["goals_scored"].fillna(0).to_numpy(), res["minutes"].fillna(0).to_numpy()
        shown["Scored?"] = np.where(goals > 1, [f"Yes ({int(g)})" for g in goals],
                                    np.where(goals > 0, "Yes", np.where(mins > 0, "No", "Didn't play")))
    st.dataframe(club_columns(shown.head(25)), hide_index=True, width="stretch", column_config={
        "Chance to score (Market Odds)": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
        "Price £m": st.column_config.NumberColumn(format="%.1f"),
        "Traded $": st.column_config.NumberColumn(format="compact")})
    if check:
        st.caption(f"Did they materialise? Across the {len(s)} traded scorer markets this gameweek, the Market "
                   f"Odds added up to {s['p_anytime'].sum():.1f} expected scorers; {int((goals > 0).sum())} "
                   "actually scored.")
    st.caption("Tested on 2025-26, they priced scorers about 60% too high "
               "and ranked them barely better than chance (AUC 0.58, against 0.73 for our own model), and "
               "most trade very little. The model only uses them to spot players who've been ruled out.")


def _render_outrights(bs: dict) -> None:
    if not markets.OUTRIGHTS_PATH.exists():
        return
    outrights = pd.read_parquet(markets.OUTRIGHTS_PATH)
    if outrights.empty:
        return
    st.markdown("**Season markets**")
    # Placeholder outcomes ("Other", "Team B") and untraded ones sit at their opening price.
    placeholder = outrights["outcome"].str.fullmatch(r"(Other|Team [A-Z]|Player [A-Z]|Manager [A-Z])", case=False)
    outrights = outrights[~placeholder & (outrights["volume"] > 0)]
    latest = outrights[outrights["snapshot"] == outrights["snapshot"].max()]
    events = latest.groupby("event")["volume"].sum().sort_values(ascending=False).index.tolist()
    ranked = sorted(events, key=lambda e: next((i for i, k in enumerate(OUTRIGHT_ORDER) if k.lower() in e.lower()),
                                               len(OUTRIGHT_ORDER)))
    event = st.selectbox("Market", ranked, key="market_outright")
    top = latest[latest["event"] == event].sort_values("probability", ascending=False).head(15)
    snapshot = _local(top["snapshot"].iloc[0])
    first_deadline = pd.Timestamp(next(e for e in bs["events"] if e["id"] == 1)["deadline_time"])
    start_at = _local(first_deadline)
    now_label, start_label = f"Now ({snapshot:%d %b})", f"Season start ({start_at:%d %b})"
    if "token" in top and top["token"].notna().any():
        start = _season_start(tuple(top["token"].dropna()), first_deadline)
        top = top.assign(start=top["token"].map(start))
    else:
        top = top.assign(start=np.nan)
    long = pd.concat([top.assign(when=start_label, chance=top["start"]),
                      top.assign(when=now_label, chance=top["probability"])], ignore_index=True).dropna(subset=["chance"])
    bars = alt.Chart(long).mark_bar().encode(
        y=alt.Y("outcome:N", sort=top["outcome"].tolist(), title=None, axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset("when:N", sort=[start_label, now_label]),
        x=alt.X("chance:Q", title="Market Odds", axis=alt.Axis(format="%")),
        color=alt.Color("when:N", title=None, sort=[start_label, now_label],
                        scale=alt.Scale(domain=[start_label, now_label], range=["#b9b8b3", MARKET]),
                        legend=alt.Legend(orient="top")),
        tooltip=["outcome", "when", alt.Tooltip("chance:Q", format=".1%"),
                 alt.Tooltip("volume:Q", format=",.0f", title="traded $")])
    st.altair_chart(bars.properties(height=alt.Step(12)), width="stretch")
    missing = int(top["start"].isna().sum())
    st.caption(f"Blue: the market now, as of the last **Fetch betting odds** ({snapshot:%a %d %b %H:%M}). "
               f"Grey: the same market at the season's first FPL deadline ({start_at:%a %d %b %H:%M}), from its "
               "price history" + (f"; {missing} outcome(s) had no price yet then." if missing else "."))


def _render_accuracy() -> None:
    acc = _accuracy(markets.MATCHES_PATH.stat().st_mtime if markets.MATCHES_PATH.exists() else 0.0)
    if acc.empty:
        return
    with st.expander("How good are the odds? Market Odds vs Our Odds, season by season"):
        shown = acc.assign(source=acc["source"].map({"Market": "Market Odds", "Our ratings": "Our Odds"}))
        st.dataframe(shown.rename(columns={"season": "Season", "source": "Source", "matches": "Matches",
                                           "goals_rmse": "Goals RMSE", "clean_sheet_brier": "Clean-sheet Brier",
                                           "win_log_loss": "Win log loss"}),
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format=DECIMAL)
                                    for c in ("Goals RMSE", "Clean-sheet Brier", "Win log loss")})
        st.caption("Lower is better in every column; each uses the odds as they stood at the FPL deadline. The "
                   "market is best at the result itself; for goals and clean sheets it's close to our own "
                   "ratings. The model uses both; so far that's close to neutral for accuracy. 2024-25 only "
                   "had result markets; goal markets started in 2025-26.")
