"""The dashboard's player profile (Players & Fixtures -> A player's season -> Where he plays): the
same view as the website's (web/src/components/Profile.tsx). His shots binned into zones round the
goal and every shot on a half pitch, his per-90 zone metrics against his position's average, and
the players whose profile is most like his (spatial.py)."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from xpfpl import config, spatial
from xpfpl.match_view import BODY, OUTCOMES, SITUATIONS
from xpfpl.style import PERCENT, POINTS, club_columns

BLUE, LINE = "#2a78d6", "#8a8984"
W, HALF = 68.0, 52.5                 # half a 105 x 68 pitch, the goal at the top


def _stamp(season: str) -> float:
    from xpfpl.data import matchstats
    files = list((matchstats.ARCHIVE / season).rglob("*.parquet"))
    return max((f.stat().st_mtime for f in files), default=0.0)


@st.cache_data(show_spinner=False)
def _profiles(season: str, data_stamp: float) -> pd.DataFrame:
    return spatial.profiles(season)


@st.cache_data(show_spinner=False)
def _shots(season: str, data_stamp: float) -> pd.DataFrame:
    return spatial.shots(season)


# Plot coordinates: x across the pitch (the shooter's left on the left: the source's across
# coordinate runs to his left, so it's mirrored), y metres from the goal line.
def _across(y):
    return W - np.asarray(y, float) / 100 * W


def _along(x):
    return np.asarray(x, float) / 100 * 105


def _half_pitch() -> alt.Chart:
    def arc(cx, cy, r, a0, a1):
        a = np.linspace(a0, a1, 41)
        return list(zip(cx + r * np.cos(a), cy + r * np.sin(a)))

    def box(depth, width):
        return [((W - width) / 2, 0), ((W - width) / 2, depth), ((W + width) / 2, depth), ((W + width) / 2, 0)]

    d = np.arccos(5.5 / 9.15)
    lines = [[(0, 0), (W, 0), (W, HALF), (0, HALF), (0, 0)], box(16.5, 40.32), box(5.5, 18.32),
             arc(W / 2, 11, 9.15, np.pi / 2 - d, np.pi / 2 + d), arc(W / 2, HALF, 9.15, np.pi, 2 * np.pi),
             [(W / 2 - 3.66, 0), (W / 2 - 3.66, -1.5), (W / 2 + 3.66, -1.5), (W / 2 + 3.66, 0)]]
    df = pd.DataFrame([{"line": i, "order": j, "px": x, "py": y} for i, pts in enumerate(lines) for j, (x, y) in enumerate(pts)])
    return alt.Chart(df).mark_line(color=LINE, strokeWidth=1.1).encode(
        x=alt.X("px:Q", scale=_X, axis=None), y=alt.Y("py:Q", scale=_Y, axis=None), detail="line:N", order="order:Q")


_X = alt.Scale(domain=[-1, W + 1], nice=False, zero=False)
_Y = alt.Scale(domain=[-2, HALF + 1], nice=False, zero=False, reverse=True)
SIZE = (380, int(380 * (HALF + 3) / (W + 2)))


def zone_chart(shots: pd.DataFrame) -> alt.LayerChart:
    table = spatial.zones()
    shots = shots.dropna(subset=["zone"])
    per = shots.groupby("zone").agg(shots=("xg", "size"), xg=("xg", "sum"), goals=("outcome", lambda o: (o == "goal").sum()))
    table = table.join(per, on="zone").fillna({"shots": 0, "xg": 0.0, "goals": 0})
    table = table.assign(px0=_across(table["y0"]), px1=_across(table["y1"]), py0=_along(table["x0"]), py1=_along(table["x1"]),
                         share=table["shots"] / max(len(shots), 1))
    table["label"] = (table["share"] * 100).round().astype(int).astype(str) + "%"
    hit = table[table["shots"] > 0]
    rects = alt.Chart(hit).mark_rect(color=BLUE, stroke="white", strokeWidth=2).encode(
        x=alt.X("px0:Q", scale=_X, axis=None), x2="px1:Q", y=alt.Y("py0:Q", scale=_Y, axis=None), y2="py1:Q",
        opacity=alt.Opacity("shots:Q", scale=alt.Scale(domain=[0, hit["shots"].max() if len(hit) else 1], range=[0.08, 0.85]), legend=None),
        tooltip=[alt.Tooltip("shots:Q", title="Shots"), alt.Tooltip("share:Q", title="Share of his shots", format=".0%"),
                 alt.Tooltip("xg:Q", title="xG", format=".2f"), alt.Tooltip("goals:Q", title="Goals")])
    text = alt.Chart(hit.assign(cx=(hit["px0"] + hit["px1"]) / 2, cy=(hit["py0"] + hit["py1"]) / 2)).mark_text(
        fontSize=11, color="#111111").encode(x=alt.X("cx:Q", scale=_X, axis=None), y=alt.Y("cy:Q", scale=_Y, axis=None), text="label:N")
    return (rects + text + _half_pitch()).properties(width=SIZE[0], height=SIZE[1])


def shot_chart(shots: pd.DataFrame, opponents: dict[int, str]) -> alt.LayerChart:
    s = shots.assign(px=_across(shots["start_y"]), py=_along(shots["start_x"]), goal=shots["outcome"].eq("goal"),
                     result=shots["outcome"].map(OUTCOMES).fillna(shots["outcome"]),
                     how=shots["situation"].map(SITUATIONS).fillna(shots["situation"]) + ", "
                     + shots["body_part"].map(BODY).fillna("").str.lower(),
                     match=[f"GW{g} v {opponents.get(f, '?')}" for g, f in zip(shots["gw"], shots["fixture"])]
                     ).sort_values("xg", ascending=False)
    x, y = alt.X("px:Q", scale=_X, axis=None), alt.Y("py:Q", scale=_Y, axis=None)
    size = alt.Size("xg:Q", scale=alt.Scale(domain=[0, 1], range=[30, 900]), legend=None)
    tip = [alt.Tooltip("match:N", title="Match"), alt.Tooltip("minute:Q", title="Minute"), alt.Tooltip("result:N", title="Outcome"),
           alt.Tooltip("xg:Q", title="xG", format=".2f"), alt.Tooltip("how:N", title="How")]
    misses = alt.Chart(s[~s["goal"]]).mark_circle(color=BLUE, fillOpacity=0.2, stroke=BLUE, strokeWidth=1.5).encode(
        x=x, y=y, size=size, tooltip=tip)
    goals = alt.Chart(s[s["goal"]]).mark_circle(color=BLUE, opacity=1, stroke="white", strokeWidth=2).encode(
        x=x, y=y, size=size, tooltip=tip)
    return (_half_pitch() + misses + goals).properties(width=SIZE[0], height=SIZE[1])


def render(season: str, element: int, bs: dict, forecast: pd.Series) -> None:
    """The profile of FPL player `element`. `forecast`: next-GW xP per element (for similar players)."""
    stamp = _stamp(season)
    table = _profiles(season, stamp)
    elements = pd.DataFrame(bs["elements"]).set_index("id")
    if table.empty or element not in elements.index:
        return
    code, pos = int(elements.at[element, "code"]), int(elements.at[element, "element_type"])
    st.markdown("**Where he plays**")
    if code not in table.index:
        st.caption("No match data for him yet this season.")
        return
    me = table.loc[code]
    position = elements.set_index("code")["element_type"]
    peers = table[(table["minutes"] >= spatial.MIN_MINUTES) & (table.index.map(position) == pos)]
    few = me["shots"] < spatial.MIN_SHOTS
    label = config.POSITIONS[pos]

    # (metric, label, format); the shot metrics need a few shots to mean anything.
    tiles = [("box_touches_p90", "Touches in the opposition box", "{:.1f} per 90"),
             ("final_third_passes_p90", "Passes into the final third", "{:.1f} per 90"),
             ("shots_p90", "Shots", "{:.1f} per 90"),
             ("box_shot_share", "Shots from inside the box", "{:.0%}"),
             ("shot_distance", "Average shot distance", "{:.1f} m"),
             ("xg_per_shot", "xG per shot", "{:.2f}")]
    for c, (metric, name, fmt) in zip(st.columns(len(tiles)), tiles):
        value, average = me.get(metric), peers[metric].mean() if metric in peers else np.nan
        c.metric(name, "-" if pd.isna(value) else fmt.format(value),
                 help=None if pd.isna(average) else f"{label} average: {fmt.format(average)}")
    if few:
        st.caption(f"Shot metrics need {spatial.MIN_SHOTS} shots; he has {me['shots']:.0f}.")
    st.caption(f"Hover a tile's (?) for the {label} average (players with {spatial.MIN_MINUTES}+ minutes).")

    mine = _shots(season, stamp)
    mine = mine[mine["code"] == code]
    if len(mine):
        teams = {t["id"]: t["short_name"] for t in bs["teams"]}
        team = int(elements.at[element, "team"])
        from xpfpl.data import archive
        fx = archive.fixtures(season).drop_duplicates("id", keep="last")
        opponents = {int(r.id): teams.get(int(r.team_a if r.team_h == team else r.team_h), "?") for r in fx.itertuples()}
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(zone_chart(mine), width="content")
            st.caption(f"His {len(mine)} shots by zone: the darker, the more came from there. Hover for xG and goals.")
        with c2:
            st.altair_chart(shot_chart(mine, opponents), width="content")
            st.caption("Every shot: the bigger, the better the chance (xG); solid circles are goals. Hover for the match.")

    found = spatial.similar(table, position).get(code, [])
    if found:
        by_code = elements.reset_index().set_index("code")
        rows = pd.DataFrame([{"Player": by_code.at[c, "web_name"], "Club": {t["id"]: t["short_name"] for t in bs["teams"]}[int(by_code.at[c, "team"])],
                              "£m": by_code.at[c, "now_cost"] / 10, "Similarity": sim,
                              "xP next GW": forecast.get(int(by_code.at[c, "id"]), np.nan)} for c, sim in found if c in by_code.index])
        st.markdown("**Similar profiles**")
        st.dataframe(club_columns(rows), hide_index=True, width="stretch",
                     column_config={"£m": st.column_config.NumberColumn(format="%.1f"),
                                    "Similarity": st.column_config.NumberColumn(format=PERCENT),
                                    "xP next GW": st.column_config.NumberColumn(format=POINTS)})
        st.caption("Players at the same position who get the ball and their shots in the most similar places (box touches, "
                   "final-third passes, chances created, shots, where they come from and how good they are, each against "
                   f"the position's average; {spatial.MIN_MINUTES}+ minutes). A similar role, not a forecast: that's the xP.")
    elif me["minutes"] < spatial.MIN_MINUTES:
        st.caption(f"Similar profiles need {spatial.MIN_MINUTES} minutes this season; he has {me['minutes']:.0f}.")
