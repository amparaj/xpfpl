"""The dashboard's Prices tab: who's about to rise or fall, your squad's selling prices, and the
season's price changes (data/pricewatch.py).

Today's numbers come from the dashboard's own bootstrap-static call, so they're as fresh as the
last "Refresh live FPL data". The change log and the last three days of progress come from the
scheduled Action's prices.json on the `prices` branch (or archive/prices/ when offline).
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from xpfpl import config
from xpfpl.data import pricewatch
from xpfpl.myteam import selling_price
from xpfpl.style import DECIMAL

RISE, FALL = "#2f8f4e", "#c4403b"
NEAR = 70          # progress (either way) from which a player is listed as close to a move


@st.cache_data(ttl=600, show_spinner="Loading the price log...")
def tracked() -> dict | None:
    return pricewatch.remote()


def likelihood(v) -> str:
    """FPL's -5..+5 likelihood as words (the sign is the direction)."""
    if v is None or pd.isna(v):
        return ""
    v = int(v)
    word = {0: "no move", 1: "unlikely", 2: "possible", 3: "likely", 4: "very likely", 5: "certain"}[min(abs(v), 5)]
    return word if v == 0 else f"{'rise' if v > 0 else 'fall'} {word}"


def bs_gw(bs: dict) -> int | None:
    return next((ev["id"] for ev in bs["events"] if ev["is_current"]), None)


def _table(now: pd.DataFrame, team_short: dict, xp: pd.Series | None) -> pd.DataFrame:
    t = now.assign(
        Team=now["team"].map(team_short), Pos=now["pos"].map(config.POSITIONS), price=now["cost"] / 10,
        gw_change=now["change_event"] / 10, season_change=now["change_start"] / 10,
        next_update=[likelihood(l[0] if len(l) else None) for l in now["lik"]],
        after_that=[likelihood(l[1] if len(l) > 1 else None) for l in now["lik"]],
        projected=[p[-1] if len(p) else np.nan for p in now["proj"]])
    if xp is not None:
        t["xp"] = t["id"].map(xp)
    return t


COLUMNS = {"name": "Player", "Team": "Team", "Pos": "Pos", "price": "£m", "pct": "Progress",
           "rate": "Per hour", "projected": "Projected", "next_update": "Next update", "after_that": "Update after",
           "net_event": "Net transfers (GW)", "selected": "Sel %", "gw_change": "GW change", "season_change": "Season change",
           "xp": "xP next GW"}


def _config(**extra) -> dict:
    return {
        "Progress": st.column_config.ProgressColumn(
            format="%.1f", min_value=-100, max_value=100,
            help="FPL's progress towards this player's next price change: +100 rises, -100 falls"),
        "Per hour": st.column_config.NumberColumn(format="%+d", help="How fast the progress is moving, per hour"),
        "Projected": st.column_config.NumberColumn(
            format="%.1f", help="FPL's projected progress at the third price update from now"),
        "Next update": st.column_config.TextColumn(help="FPL's likelihood of a move at the next price update"),
        "Update after": st.column_config.TextColumn(help="... and at the one after"),
        "Net transfers (GW)": st.column_config.NumberColumn(format="%+d"),
        "Sel %": st.column_config.NumberColumn(format=DECIMAL),
        "£m": st.column_config.NumberColumn(format="%.1f"),
        "GW change": st.column_config.NumberColumn(format="%+.1f"),
        "Season change": st.column_config.NumberColumn(format="%+.1f"),
        "xP next GW": st.column_config.NumberColumn(format="%.2f"),
        **extra,
    }


def _shown(t: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return t[[c for c in cols if c in t]].rename(columns=COLUMNS)


def render(bs: dict, team_short: dict, me=None, xp: pd.Series | None = None) -> None:
    """`me`: myteam.MyTeam (or None); `xp`: next gameweek's xP by element id (or None)."""
    now = pricewatch.players(bs)
    if now["pct"].isna().all():
        st.info("FPL isn't publishing price-change progress right now (it's new in 2026-27), so only the "
                "moves below are shown.")
    t = _table(now, team_short, xp)
    state = tracked()

    rises, falls = int((now["change_event"] > 0).sum()), int((now["change_event"] < 0).sum())
    c = st.columns(4)
    c[0].metric("Rises this gameweek", rises)
    c[1].metric("Falls this gameweek", falls)
    c[2].metric(f"Close to a rise ({NEAR}+)", int((now["pct"] >= NEAR).sum()))
    c[3].metric(f"Close to a fall ({NEAR}+)", int((now["pct"] <= -NEAR).sum()))
    st.caption("Progress is FPL's own figure (bootstrap-static, new in 2026-27): a player rises at +100 and falls at "
               "-100, at one of FPL's price updates (usually overnight UK time). 'Per hour' is how fast it's moving, "
               "'Next update' and 'Update after' FPL's likelihood of a move at the coming updates. This tab is as "
               "fresh as the last 'Refresh live FPL data'.")

    if me is not None:
        st.markdown("**Your squad**")
        squad = t[t["id"].isin(list(me.squad))].copy()
        squad["bought"] = squad["id"].map(me.purchase) / 10
        squad["selling"] = squad["id"].map(me.squad)
        squad["if_rise"] = [selling_price(c + 1, me.purchase.get(i, c)) - s
                            for i, c, s in zip(squad["id"], squad["cost"], squad["selling"])]
        squad["if_fall"] = [selling_price(c - 1, me.purchase.get(i, c)) - s
                            for i, c, s in zip(squad["id"], squad["cost"], squad["selling"])]
        squad = squad.sort_values("pct", ascending=False)
        shown = _shown(squad, ["name", "Team", "Pos", "price", "bought", "selling", "if_rise", "if_fall", "pct", "rate",
                               "next_update", "after_that", "net_event", "selected", "gw_change"]).rename(
            columns={"bought": "Bought", "selling": "Selling", "if_rise": "Selling if he rises",
                     "if_fall": "Selling if he falls"})
        st.dataframe(shown, hide_index=True, width="stretch", column_config=_config(**{
            c: st.column_config.NumberColumn(format="%.1f") for c in ("Bought", "Selling")}, **{
            c: st.column_config.NumberColumn(format="%+.1f") for c in ("Selling if he rises", "Selling if he falls")}))
        value = sum(me.squad.values())
        st.caption(f"Squad selling value £{value:.1f}m + £{me.bank:.1f}m in the bank. You keep half of a rise (rounded "
                   "down to £0.1m), so a rise only adds to the selling price every second time; a fall below what "
                   "you paid costs the full £0.1m.")

    cols = ["name", "Team", "Pos", "price", "pct", "rate", "projected", "next_update", "after_that",
            "net_event", "selected", "gw_change", "xp"]
    left, right = st.columns(2)
    with left:
        st.markdown("**Closest to a rise**")
        st.dataframe(_shown(t[t["pct"] > 0].sort_values("pct", ascending=False).head(25), cols),
                     hide_index=True, width="stretch", height=420, column_config=_config())
    with right:
        st.markdown("**Closest to a fall**")
        st.dataframe(_shown(t[t["pct"] < 0].sort_values("pct").head(25), cols),
                     hide_index=True, width="stretch", height=420, column_config=_config())

    st.markdown("**Price changes**")
    log = pricewatch.change_log(state)
    if log.empty:
        season = state["season"] if state else None
        log = pricewatch.archived_changes(season) if season else log
    if log.empty:
        moved = now[now["change_event"] != 0]
        log = pd.DataFrame({"t": pd.NaT, "gw": bs_gw(bs), "id": moved["id"], "code": moved["code"], "name": moved["name"],
                            "team_code": moved["team_code"], "pos": moved["pos"],
                            "from": moved["cost"] - moved["change_event"], "to": moved["cost"]})
        st.caption("The hourly price log isn't available (the Action hasn't run yet, or you're offline): these are "
                   "this gameweek's moves from FPL, without times.")
    code_team = {t_["code"]: t_["id"] for t_ in bs["teams"]}
    log["t"] = pd.to_datetime(log["t"], utc=True)
    log = log.assign(when=log["t"].dt.tz_convert("Europe/London").dt.strftime("%a %d %b"),
                     Team=log["team_code"].map(code_team).map(team_short), Pos=log["pos"].map(config.POSITIONS),
                     move=np.where(log["to"] > log["from"], "Rise", "Fall"),
                     price_from=log["from"] / 10, price_to=log["to"] / 10)
    log["when"] = log["when"].where(log["t"].notna(), log["gw"].map(lambda g: f"GW{int(g)}" if pd.notna(g) else ""))
    filt = st.segmented_control("Show", ["All", "Rises", "Falls"], default="All", key="price_log_filter")
    if filt == "Rises":
        log = log[log["move"] == "Rise"]
    elif filt == "Falls":
        log = log[log["move"] == "Fall"]
    st.dataframe(log[["when", "name", "Team", "Pos", "move", "price_from", "price_to"]].rename(columns={
        "when": "Day (UK)", "name": "Player", "move": "Move", "price_from": "From £m", "price_to": "To £m"}),
        hide_index=True, width="stretch", height=320,
        column_config={"From £m": st.column_config.NumberColumn(format="%.1f"),
                       "To £m": st.column_config.NumberColumn(format="%.1f")})
    if state:
        st.caption(f"Logged by the hourly price Action (last run {pd.Timestamp(state['updated']).tz_convert('Europe/London'):%a %d %b %H:%M} UK). "
                   "A move's day is the run that first saw it; moves without a day were made before the log started.")

    st.markdown("**A player's progress, last three days**")
    if not state or not state.get("times"):
        st.caption("Needs the hourly price Action's log.")
        return
    label = {r.id: f"{r.name} ({team_short.get(r.team, '')}, £{r.cost / 10:.1f})" for r in now.itertuples()}
    tracked_players = {p["id"]: p for p in state["players"]}
    default = int(t.loc[t["pct"].abs().idxmax(), "id"]) if t["pct"].notna().any() else int(t["id"].iloc[0])
    options = sorted(label, key=lambda i: label[i])
    chosen = st.selectbox("Player", options, index=options.index(default) if default in options else 0,
                          format_func=lambda i: label[i], key="price_player")
    p = tracked_players.get(chosen)
    if p is None:
        st.caption("Not in the log yet.")
        return
    series = pd.DataFrame({"time": pd.to_datetime(state["times"], utc=True).tz_convert("Europe/London").tz_localize(None),
                           "progress": p["h"]}).dropna()
    rule = pd.DataFrame({"y": [100, -100]})
    chart = (alt.Chart(series).mark_line(point=True, color="#2a78d6").encode(
                x=alt.X("time:T", title="UK time"),
                y=alt.Y("progress:Q", title="Progress", scale=alt.Scale(domain=[-100, 100])),
                tooltip=[alt.Tooltip("time:T", title="Time", format="%a %d %b %H:%M"),
                         alt.Tooltip("progress:Q", title="Progress", format=".1f")])
             + alt.Chart(rule).mark_rule(strokeDash=[4, 4], color="#8a8984").encode(y="y:Q"))
    st.altair_chart(chart, width="stretch")
    st.caption("Dashed lines: +100 (rise) and -100 (fall). The progress usually resets to near 0 after a move.")
