"""The dashboard's match view (Gameweek Review -> Matches): one played match as it happened.

The same views as the website's match page (web/src/pages/Match.tsx): a shot map (hover a shot for
the player, minute, outcome, xG and how the chance came about), where the shots on target went,
expected goals and momentum through the match, the team stats, the odds before the deadline, and
every player's FPL points next to their xP and match stats. The match data is FotMob's, via
FPL-Core-Insights (data/matchstats.py); `xpfpl fetch` keeps it up to date.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from xpfpl import config, market_view
from xpfpl.data import archive, markets, matchstats
from xpfpl.style import DECIMAL, PERCENT, POINTS, club_columns

HOME, AWAY = "#2a78d6", "#eb6834"      # the site's --s1 / --s2
LINE = "#8a8984"
LENGTH, WIDTH = 105.0, 68.0             # metres; the source's coordinates are 0-100 both ways
OUTCOMES = {"goal": "Goal", "save": "Saved", "miss": "Off target", "block": "Blocked", "post": "Hit the woodwork",
            "blocked-off-line": "Cleared off the line", "own-goal": "Own goal"}
SITUATIONS = {"assisted": "Open play, assisted", "regular": "Open play", "fast-break": "Counter-attack",
              "corner": "From a corner", "set-piece": "Set piece", "throw-in-set-piece": "From a throw-in",
              "free-kick": "Direct free kick", "penalty": "Penalty"}
BODY = {"right-foot": "Right foot", "left-foot": "Left foot", "head": "Header", "other": "Other"}
TEAM_STATS = [("possession", "Possession (%)", 0), ("field_tilt", "Field tilt (%)", 0), ("expected_goals_xg", "Expected goals (xG)", 2),
              ("xg_open_play", "xG from open play", 2), ("xg_set_play", "xG from set pieces", 2),
              ("xg_on_target_xgot", "xG on target (xGOT)", 2), ("total_shots", "Shots", 0),
              ("shots_on_target", "Shots on target", 0), ("big_chances", "Big chances", 0),
              ("big_chances_missed", "Big chances missed", 0), ("hit_woodwork", "Hit the woodwork", 0),
              ("touches_in_opposition_box", "Touches in the opposition box", 0), ("passes", "Passes", 0),
              ("accurate_passes_pct", "Pass accuracy (%)", 0), ("corners", "Corners", 0), ("keeper_saves", "Saves", 0),
              ("tackles_won", "Tackles won", 0), ("interceptions", "Interceptions", 0), ("clearances", "Clearances", 0),
              ("duels_won", "Duels won", 0), ("fouls_committed", "Fouls", 0), ("offsides", "Offsides", 0),
              ("yellow_cards", "Yellow cards", 0), ("red_cards", "Red cards", 0)]


def _stamp(season: str) -> float:
    """Changes whenever the archived match data does (the cache key)."""
    files = list((matchstats.ARCHIVE / season).rglob("*.parquet"))
    return max((f.stat().st_mtime for f in files), default=0.0)


@st.cache_data(show_spinner=False)
def _gameweek(season: str, gw: int, data_stamp: float) -> dict[str, pd.DataFrame]:
    return matchstats.gameweek(season, gw)


@st.cache_data(show_spinner=False)
def _fpl_rows(season: str, gw: int, data_stamp: float) -> pd.DataFrame:
    """FPL's per-match rows for the gameweek (points, minutes, bonus...)."""
    rows, _ = archive.season_tables(season)
    return rows[rows["round"] == gw] if len(rows) else rows


# ---------------------------------------------------------------- shots

def _shots(raw: pd.DataFrame, names: dict[int, str], home: str, away: str) -> pd.DataFrame:
    """Shots in pitch metres (the home side attacking to the right), with labels for the tooltips."""
    s = raw.sort_values(["minute", "added_time", "shot_index"], na_position="first").copy()
    home_side = s["is_home"].astype(bool)
    along, across = LENGTH * (1 - s["start_x"] / 100), WIDTH * s["start_y"] / 100
    added = s["added_time"].fillna(0)
    s = s.assign(
        side=np.where(home_side, "home", "away"), club=np.where(home_side, home, away),
        player=s["code"].map(names).fillna("Player not recorded"),
        x=np.where(home_side, along, LENGTH - along), y=np.where(home_side, across, WIDTH - across),
        clock=[f"{m}+{a:.0f}'" if a else f"{m}'" for m, a in zip(s["minute"], added)],
        # Keeps first-half stoppage time before the second half on the time axis.
        t=s["minute"] + np.where(s["minute"] == 45, added.clip(upper=4) * 0.2, added),
        result=s["outcome"].map(OUTCOMES).fillna(s["outcome"]),
        how=s["situation"].map(SITUATIONS).fillna(s["situation"]) + ", " + s["body_part"].map(BODY).fillna("").str.lower(),
        goal=s["outcome"].eq("goal"),
    )
    s["running_xg"] = s.groupby("side")["xg"].cumsum()
    return s


TOOLTIP = [alt.Tooltip("player:N", title="Player"), alt.Tooltip("club:N", title="Club"),
           alt.Tooltip("clock:N", title="Minute"), alt.Tooltip("result:N", title="Outcome"),
           alt.Tooltip("xg:Q", title="xG", format=".2f"), alt.Tooltip("xgot:Q", title="xG on target", format=".2f"),
           alt.Tooltip("how:N", title="How")]


def _pitch_lines() -> pd.DataFrame:
    """Pitch markings in metres, as one row per point of each line (`line` groups them)."""
    def arc(cx, cy, r, a0, a1):
        a = np.linspace(a0, a1, 41)
        return list(zip(cx + r * np.cos(a), cy + r * np.sin(a)))

    def box(x0, depth, width, d):
        y0, y1 = (WIDTH - width) / 2, (WIDTH + width) / 2
        return [(x0, y0), (x0 + d * depth, y0), (x0 + d * depth, y1), (x0, y1)]

    d = np.arccos(5.5 / 9.15)            # the part of the circle round the spot outside the box
    g0, g1 = WIDTH / 2 - 3.66, WIDTH / 2 + 3.66
    lines = [[(0, 0), (LENGTH, 0), (LENGTH, WIDTH), (0, WIDTH), (0, 0)], [(LENGTH / 2, 0), (LENGTH / 2, WIDTH)],
             arc(LENGTH / 2, WIDTH / 2, 9.15, 0, 2 * np.pi), box(0, 16.5, 40.32, 1), box(LENGTH, 16.5, 40.32, -1),
             box(0, 5.5, 18.32, 1), box(LENGTH, 5.5, 18.32, -1),
             arc(11, WIDTH / 2, 9.15, -d, d), arc(LENGTH - 11, WIDTH / 2, 9.15, np.pi - d, np.pi + d),
             [(0, g0), (-1.5, g0), (-1.5, g1), (0, g1)], [(LENGTH, g0), (LENGTH + 1.5, g0), (LENGTH + 1.5, g1), (LENGTH, g1)]]
    return pd.DataFrame([{"line": i, "order": j, "x": x, "y": y} for i, pts in enumerate(lines) for j, (x, y) in enumerate(pts)])


def _side_colour() -> alt.Color:
    return alt.Color("side:N", scale=alt.Scale(domain=["home", "away"], range=[HOME, AWAY]), legend=None)


def shot_map(shots: pd.DataFrame, width: int = 760) -> alt.LayerChart:
    x = alt.X("x:Q", scale=alt.Scale(domain=[-2, LENGTH + 2], nice=False, zero=False), axis=None)
    y = alt.Y("y:Q", scale=alt.Scale(domain=[-2, WIDTH + 2], nice=False, zero=False), axis=None)
    size = alt.Size("xg:Q", scale=alt.Scale(domain=[0, 1], range=[40, 1600]), legend=None)
    pitch = alt.Chart(_pitch_lines()).mark_line(color=LINE, strokeWidth=1.2).encode(x=x, y=y, detail="line:N", order="order:Q")
    spots = alt.Chart(pd.DataFrame({"x": [11, LENGTH - 11, LENGTH / 2], "y": [WIDTH / 2] * 3})).mark_circle(
        color=LINE, size=12).encode(x=x, y=y)
    # Biggest chances drawn first, so the small ones sit on top and stay hoverable.
    ordered = shots.sort_values("xg", ascending=False)
    misses = alt.Chart(ordered[~ordered["goal"]]).mark_circle(fillOpacity=0.2, strokeOpacity=1, strokeWidth=1.5).encode(
        x=x, y=y, size=size, color=_side_colour(), stroke=_side_colour(), tooltip=TOOLTIP)
    goals = alt.Chart(ordered[ordered["goal"]]).mark_circle(opacity=1, stroke="white", strokeWidth=2).encode(
        x=x, y=y, size=size, color=_side_colour(), tooltip=TOOLTIP)
    scorers = ordered[ordered["goal"]].drop_duplicates("player")        # each scorer named once
    labels = alt.Chart(scorers).mark_text(dy=-16, fontSize=11, color="#333333").encode(x=x, y=y, text="player:N")
    return (pitch + spots + misses + goals + labels).properties(
        width=width, height=int(width * (WIDTH + 4) / (LENGTH + 4)))


def on_goal(shots: pd.DataFrame, width: int = 380) -> alt.LayerChart | None:
    """Shots that reached the goal on the goal frame, as the shooter sees it (the source's
    goal-mouth y runs to the shooter's left)."""
    reached = shots[shots["outcome"].isin(["goal", "save", "post", "blocked-off-line"])].assign(
        across=lambda d: 100 - d["goal_mouth_y"], size_by=lambda d: d["xgot"].fillna(d["xg"]))
    if reached.empty:
        return None
    x = alt.X("across:Q", scale=alt.Scale(domain=[43, 57], nice=False, zero=False), axis=None)
    y = alt.Y("goal_mouth_z:Q", scale=alt.Scale(domain=[-2, 44], nice=False, zero=False), axis=None)
    size = alt.Size("size_by:Q", scale=alt.Scale(domain=[0, 1], range=[40, 700]), legend=None)
    frame = pd.DataFrame({"across": [44.8, 44.8, 55.2, 55.2], "goal_mouth_z": [0, 38, 38, 0], "order": range(4)})
    ground = pd.DataFrame({"across": [43, 57], "goal_mouth_z": [0, 0], "order": [0, 1]})
    post = alt.Chart(frame).mark_line(color="#52514e", strokeWidth=3).encode(x=x, y=y, order="order:Q")
    line = alt.Chart(ground).mark_line(color=LINE).encode(x=x, y=y, order="order:Q")
    saved = alt.Chart(reached[~reached["goal"]]).mark_circle(fillOpacity=0.2, strokeOpacity=1, strokeWidth=1.5).encode(
        x=x, y=y, size=size, color=_side_colour(), stroke=_side_colour(), tooltip=TOOLTIP)
    scored = alt.Chart(reached[reached["goal"]]).mark_circle(opacity=1, stroke="white", strokeWidth=2).encode(
        x=x, y=y, size=size, color=_side_colour(), tooltip=TOOLTIP)
    return (line + post + saved + scored).properties(width=width, height=int(width * 0.5))


def xg_race(shots: pd.DataFrame, end: float) -> alt.LayerChart:
    """Running xG for each side, a step at each shot, goals marked."""
    steps = []
    for side in ("home", "away"):
        mine = shots[shots["side"] == side]
        last = float(mine["running_xg"].iloc[-1]) if len(mine) else 0.0
        steps.append(pd.concat([pd.DataFrame({"t": [0.0], "running_xg": [0.0], "side": [side]}),
                                mine[["t", "running_xg", "side"]],
                                pd.DataFrame({"t": [end], "running_xg": [last], "side": [side]})], ignore_index=True))
    x = alt.X("t:Q", title="Minute", scale=alt.Scale(domain=[0, end], nice=False), axis=alt.Axis(values=list(range(0, 91, 15))))
    y = alt.Y("running_xg:Q", title="Expected goals so far")
    lines = alt.Chart(pd.concat(steps)).mark_line(interpolate="step-after", strokeWidth=2).encode(
        x=x, y=y, color=_side_colour(), detail="side:N")
    half = alt.Chart(pd.DataFrame({"t": [45]})).mark_rule(color="#cccccc", strokeDash=[3, 3]).encode(x="t:Q")
    hover = alt.Chart(shots).mark_circle(size=60, opacity=0.01).encode(
        x=x, y=y, tooltip=TOOLTIP + [alt.Tooltip("running_xg:Q", title="Club xG so far", format=".2f")])
    goals = shots[shots["goal"]].assign(label=lambda d: d["player"] + " " + d["clock"])
    dots = alt.Chart(goals).mark_circle(size=80, opacity=1, stroke="white", strokeWidth=2).encode(
        x=x, y=y, color=_side_colour(), tooltip=TOOLTIP)
    text = alt.Chart(goals).mark_text(dy=-12, fontSize=11, color="#333333").encode(x=x, y=y, text="label:N")
    return (half + lines + hover + dots + text).properties(height=260)


def momentum_chart(flow: pd.DataFrame, shots: pd.DataFrame, end: float, home: str, away: str) -> alt.LayerChart:
    """FotMob's momentum per minute: up for the home side, down for the away side."""
    flow = flow.assign(x0=flow["minute"] - 0.45, x1=flow["minute"] + 0.45, zero=0.0,
                       side=np.where(flow["value"] >= 0, "home", "away"),
                       on_top=np.where(flow["value"] > 0, home, np.where(flow["value"] < 0, away, "Even")))
    x = alt.X("x0:Q", title="Minute", scale=alt.Scale(domain=[0, end], nice=False), axis=alt.Axis(values=list(range(0, 91, 15))))
    bars = alt.Chart(flow).mark_rect().encode(
        x=x, x2="x1:Q", y=alt.Y("zero:Q", title=f"↓ {away} · {home} ↑", scale=alt.Scale(domain=[-100, 100]),
                                   axis=alt.Axis(labels=False, ticks=False)),
        y2="value:Q", color=_side_colour(),
        tooltip=[alt.Tooltip("minute:Q", title="Minute"), alt.Tooltip("on_top:N", title="On top"),
                 alt.Tooltip("value:Q", title="Momentum")])
    goals = shots[shots["goal"]].assign(x0=lambda d: d["t"], level=lambda d: np.where(d["side"] == "home", 92, -92))
    dots = alt.Chart(goals).mark_circle(size=70, opacity=1, stroke="white", strokeWidth=2).encode(
        x="x0:Q", y="level:Q", color=_side_colour(), tooltip=TOOLTIP)
    return (bars + dots).properties(height=260)


def team_stats_chart(stats: pd.DataFrame, home: str, away: str) -> alt.Chart:
    """Each stat as a bar split by the two sides' shares, the values in the label."""
    by = stats.set_index("stat")[["home", "away"]].copy()
    # Field tilt: each side's share of the passes played in the opposition half.
    if "opposition_half" in by.index and by.loc["opposition_half"].notna().all() and by.loc["opposition_half"].sum() > 0:
        by.loc["field_tilt"] = 100 * by.loc["opposition_half"] / by.loc["opposition_half"].sum()
    rows = []
    for key, label, digits in TEAM_STATS:
        if key not in by.index or by.loc[key, ["home", "away"]].isna().any():
            continue
        h, a = float(by.at[key, "home"]), float(by.at[key, "away"])
        if h + a == 0:          # 0-0 (no offsides, no red cards): nothing to split
            continue
        name = f"{h:.{digits}f}   {label}   {a:.{digits}f}"
        share = h / (h + a)
        rows += [{"stat": name, "side": "home", "share": share, "club": home, "value": h, "order": 0},
                 {"stat": name, "side": "away", "share": 1 - share, "club": away, "value": a, "order": 1}]
    table = pd.DataFrame(rows)
    order = list(dict.fromkeys(table["stat"]))
    return alt.Chart(table).mark_bar(size=12).encode(
        y=alt.Y("stat:N", sort=order, title=None, axis=alt.Axis(labelLimit=400, labelFontSize=12, ticks=False, domain=False)),
        x=alt.X("share:Q", stack="normalize", axis=None), color=_side_colour(), order="order:Q",
        tooltip=[alt.Tooltip("club:N", title="Club"), alt.Tooltip("value:Q", title="Value")],
    ).properties(height=26 * len(order))


# ---------------------------------------------------------------- the view

def render(season: str, gw: int, bs: dict, fixtures: list[dict], xp: pd.Series) -> None:
    """The Matches section of Gameweek Review, for gameweek `gw`. `xp`: the model's xP per element."""
    teams = {t["id"]: t for t in bs["teams"]}
    played = [f for f in fixtures if f["event"] == gw and f["team_h_score"] is not None]
    if not played:
        return
    stamp = _stamp(season)
    data = _gameweek(season, gw, stamp)
    label = {f["id"]: f"{teams[f['team_h']]['name']} {f['team_h_score']}–{f['team_a_score']} {teams[f['team_a']]['name']}"
             for f in played}
    st.markdown(f"**Matches in GW{gw}**")
    fid = st.selectbox("Match", list(label), format_func=label.get, key=f"match_gw{gw}",
                       help="Shots, how the match went, the team stats, the odds before the deadline and every player's numbers")
    f = next(x for x in played if x["id"] == fid)
    home, away = teams[f["team_h"]]["short_name"], teams[f["team_a"]]["short_name"]
    part = {k: df[df["fixture"] == fid] for k, df in data.items()}
    if "shots" not in part or part["shots"].empty:
        st.info("No shot data for this match yet: it arrives a day or two after the match, with the next "
                "**Fetch match data**.")
    else:
        elements = pd.DataFrame(bs["elements"])
        names = dict(zip(elements["code"], elements["web_name"]))
        shots = _shots(part["shots"], names, home, away)
        stats = part.get("matches", pd.DataFrame(columns=["stat", "home", "away"])).set_index("stat")
        stat = lambda k: stats.loc[k, ["home", "away"]].tolist() if k in stats.index else [np.nan, np.nan]  # noqa: E731
        m = st.columns(4)
        (xh, xa), (sh, sa), (oh, oa), (bh, ba) = (stat(k) for k in ("expected_goals_xg", "total_shots",
                                                                    "shots_on_target", "big_chances"))
        m[0].metric("Expected goals", f"{xh:.2f} – {xa:.2f}", help="The quality of the chances each side had")
        m[1].metric("Shots (on target)", f"{sh:.0f} ({oh:.0f}) – {sa:.0f} ({oa:.0f})")
        m[2].metric("Big chances", f"{bh:.0f} – {ba:.0f}")
        m[3].metric("Possession", "{:.0f}% – {:.0f}%".format(*stat("possession")))

        st.markdown(f"**Shot map**: :blue[●] {home} attacking → · :orange[●] {away} ← attacking")
        side = st.segmented_control("Shots", ["Both", home, away], default="Both", key=f"shots_{fid}",
                                    label_visibility="collapsed") or "Both"
        shown = shots if side == "Both" else shots[shots["club"] == side]
        st.altair_chart(shot_map(shown), width="content")
        st.caption("One circle per shot, where it was taken; the bigger, the better the chance (xG). Solid circles are "
                   "goals. Hover a circle for the player, minute, outcome and how the chance came about.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Where the shots on target went**")
            chart = on_goal(shots)
            if chart is None:
                st.caption("No shots on target.")
            else:
                st.altair_chart(chart, width="content")
                st.caption("Both sides' shots on the same goal, as the shooter sees it. Size is xG on target; "
                           "solid circles are goals.")
        with c2:
            st.markdown("**How the match went**")
            flow = part.get("momentum", pd.DataFrame())
            end = float(max(95, shots["t"].max() + 1, flow["minute"].max() + 1 if len(flow) else 0))
            view = st.segmented_control("Chart", ["Expected goals", "Momentum"] if len(flow) else ["Expected goals"],
                                        default="Expected goals", key=f"flow_{fid}", label_visibility="collapsed")
            if view == "Momentum":
                st.altair_chart(momentum_chart(flow, shots, end, home, away), width="stretch")
                st.caption(f"FotMob's momentum: which side was on top each minute (up: {home}, down: {away}). Dots are goals.")
            else:
                st.altair_chart(xg_race(shots, end), width="stretch")
                st.caption("Each step is a shot, as big as its xG; dots are goals. Hover for the shot.")

        if len(stats):
            with st.expander("Team stats", expanded=True):
                st.altair_chart(team_stats_chart(stats.reset_index(), home, away), width="stretch")
                st.caption("Field tilt: each side's share of the passes played in the opposition half, i.e. how much "
                           "of the match was played in its attacking half.")

    _odds(season, gw, f, home, away)
    _players(season, gw, f, part.get("players"), xp, bs, stamp)
    st.caption("Match data: FotMob, via FPL-Core-Insights. About one shot in eight has no player recorded by the source.")


def _odds(season: str, gw: int, f: dict, home: str, away: str) -> None:
    market, _ = markets.load()
    if market is None:
        return
    row = market[(market["season"] == season) & (market["fixture"] == f["id"])]
    if row.empty:
        return
    row = market_view._with_ours(row, season, gw).iloc[0]
    happened = 0 if f["team_h_score"] > f["team_a_score"] else 1 if f["team_h_score"] == f["team_a_score"] else 2
    table = pd.DataFrame([
        {"Source": "Market Odds (Polymarket at the FPL deadline)", "p": [row["home_win"], row["draw"], row["away_win"]]},
        {"Source": "Our Odds (the model's club ratings)", "p": [row["ours_home_win"], row["ours_draw"], row["ours_away_win"]]}])
    outcomes = [f"{home} win", "Draw", f"{away} win"]
    outcomes[happened] += " ✓"
    for i, name in enumerate(outcomes):
        table[name] = table["p"].map(lambda p: p[i])
    table["Log loss"] = table["p"].map(lambda p: -np.log(max(p[happened], 1e-3)) if pd.notna(p[happened]) else np.nan)
    st.markdown("**What the odds said before the deadline**")
    st.dataframe(table.drop(columns="p"), hide_index=True, width="stretch",
                 column_config={**{o: st.column_config.NumberColumn(format=PERCENT) for o in outcomes},
                                "Log loss": st.column_config.NumberColumn(format="%.3f")})
    st.caption("✓ marks what happened. Log loss scores a forecast on the chance it gave to that result only (lower is "
               "better; a third each scores 1.099). One match says little: the Markets tab has the season's scores.")


def _players(season: str, gw: int, f: dict, stats: pd.DataFrame | None, xp: pd.Series, bs: dict, stamp: float) -> None:
    rows = _fpl_rows(season, gw, stamp)
    if rows.empty:
        return
    mine = rows[(rows["fixture"] == f["id"]) & (rows["minutes"] > 0)]
    if mine.empty:
        return
    elements = pd.DataFrame(bs["elements"]).set_index("id")
    short = {t["id"]: t["short_name"] for t in bs["teams"]}
    double = rows["element"].value_counts().reindex(mine["element"]).gt(1).any()
    table = pd.DataFrame({
        "Player": mine["element"].map(elements["web_name"]).to_numpy(),
        "Club": mine["element"].map(elements["team"]).map(short).to_numpy(),
        "Pos": mine["element"].map(elements["element_type"]).map(config.POSITIONS).to_numpy(),
        "Mins": mine["minutes"].to_numpy(), "Points": mine["total_points"].to_numpy(),
        "xP": mine["element"].map(xp).to_numpy(), "Bonus": mine["bonus"].to_numpy(), "BPS": mine["bps"].to_numpy(),
    })
    table["Points − xP"] = np.nan if double else table["Points"] - table["xP"]
    if stats is not None and len(stats):
        by_code = stats.set_index("code")
        codes = mine["element"].map(elements["code"]).to_numpy()
        for col, name in (("total_shots", "Shots"), ("xg", "xG"), ("xa", "xA"), ("chances_created", "Chances created"),
                          ("touches_opposition_box", "Box touches"), ("touches", "Touches"),
                          ("accurate_passes_percent", "Pass %"), ("tackles", "Tackles"),
                          ("defensive_contributions", "DC"), ("distance_covered", "Distance (km)")):
            if col in by_code:
                table[name] = by_code[col].reindex(codes).to_numpy()
    table = table.sort_values(["Points", "xP"], ascending=False)
    st.markdown("**Players**")
    st.dataframe(club_columns(table), hide_index=True, width="stretch",
                 column_config={"xP": st.column_config.NumberColumn(format=POINTS),
                                "Points − xP": st.column_config.NumberColumn(format="%+.2f"),
                                **{c: st.column_config.NumberColumn(format=DECIMAL)
                                   for c in ("xG", "xA", "Distance (km)")},
                                "Pass %": st.column_config.NumberColumn(format="%.0f%%")})
    st.caption("Everyone who played. Mins to BPS are FPL's own; the rest are the match data's."
               + (" In a double gameweek xP covers both matches, so Points − xP is left blank." if double else ""))
