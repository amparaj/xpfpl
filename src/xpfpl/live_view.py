"""The dashboard's Live Gameweek tab: the gameweek in progress (or just played) against the
forecast saved before its deadline (live.py): both teams live, where the misses came from,
who beat the forecast or fell short and why, and what to look at before the next deadline.
The website's Live Gameweek page shows the same."""

import pandas as pd
import streamlit as st

from xpfpl import config, live, scorecard, style
from xpfpl.data import api

PARTS = {"minutes": "Minutes", "finishing": "Finishing", "other": "Chances & the rest"}
LESSONS = {
    "backed": ("The chances back it up", "Beat their xP by 3+ with xG + xA of 0.6 or more: worth a look."),
    "lucky": ("Returns from little", "Beat their xP by 3+ mostly through finishing, from under 0.3 xG + xA: don't chase on this alone."),
    "unlucky": ("Chances without the points", "0.7+ xG + xA in 60+ minutes and 3 points or fewer."),
    "minutes": ("Watch their place", "Fit at the deadline and given 80%+ to play, but didn't start."),
    "stale": ("News changed after the forecast", "Their flag at the deadline wasn't the one the forecast was made with."),
}


STATES = {live.LIVE: "Playing", live.TO_PLAY: "To play", live.BLANK: "No match"}


def status(data: dict) -> tuple[str, str]:
    """Where the gameweek is, as a badge: red while a match is on, orange in between and while FPL adds
    the bonus and checks the data, green once it's confirmed."""
    states = [m["state"] for m in data["matches"]]
    if live.LIVE in states:
        return f"Live: {states.count(live.LIVE)} playing now", "red"
    if live.TO_PLAY in states:
        return "In progress: more matches to come", "orange"
    return {"complete": ("Complete", "green"), "checking": ("Checking: FPL confirming the points", "orange")}.get(
        data.get("phase"), ("Matches over: bonus to come", "orange"))


@st.cache_data(ttl=120, show_spinner="Loading the live gameweek...")
def analysis(_bs: dict, _fx: list[dict], gw: int | None, model: str) -> dict | None:
    """live.local, cached two minutes (`gw` keys the cache)."""
    return live.local(_bs, _fx, model)


def _parts(summary: dict, label: str) -> dict:
    return {"Group": label, "Players": summary.get("n", 0), "xP": summary.get("xp"), "Points": summary.get("points"),
            "Points − xP": summary.get("bias"), **{v: summary.get(k) for k, v in PARTS.items()}}


def render(bs: dict, fx: list[dict], team_short: dict[int, str], model: str = config.MODEL) -> None:
    current = next((e["id"] for e in bs["events"] if e["is_current"]), None)
    data = analysis(bs, fx, current, model) if current else None
    if data is None:
        st.info("No gameweek in progress with a saved forecast. From the next deadline this tab follows that "
                "gameweek against the forecast saved before it.")
        return
    gw = data["gw"]
    players = pd.DataFrame(bs["elements"]).set_index("id")
    t = pd.DataFrame(data["players"]).set_index("element")
    t["Player"] = players["web_name"].reindex(t.index)
    t["Club"] = players["team"].reindex(t.index).map(team_short)
    t["Pos"] = players["element_type"].reindex(t.index).map(config.POSITIONS)
    t["Why"] = t["reasons"].map(lambda r: " · ".join(r))
    name = lambda e: players.at[e, "web_name"] if e in players.index else str(e)   # noqa: E731

    label, colour = status(data)
    st.subheader(f"GW{gw}: {data['done']} of {data['total']} matches over")
    st.badge(label, color=colour)
    st.caption(f"Against the {data['model']} forecast saved before the deadline"
               + (f" ({data['forecast_made'][:16].replace('T', ' ')} UTC)" if data.get("forecast_made") else "")
               + f". Live points from FPL, as of {data['generated'][:16].replace('T', ' ')} UTC (cached two minutes; "
               "**Refresh live FPL data** reloads now). The next gameweek is planned in Plan Ahead.")

    cols = st.columns(4)
    for col, (key, label) in zip(cols, (("model", "This Model's Team"), ("mine", "My Team"))):
        team = data["teams"].get(key)
        if team:
            delta = None if team["forecast"] is None else team["points"] - team["forecast"]
            col.metric(label, f"{team['points']:.0f} pts", None if delta is None else f"{delta:+.0f} vs forecast {team['forecast']:.0f}",
                       help=(f"Better than {team['percentile']:.0%} of its simulated weeks." if team["settled"] and team["percentile"] is not None
                             else f"{team['still_to_play']} still to play ({team['still_xp']:.1f} xP to come)."))
    week = data["week"]
    if week.get("n"):
        cols[2].metric("Players expected to feature", f"{week['bias']:+.2f} a player",
                       help=f"{week['points']:.2f} points each against {week['xp']:.2f} xP, {week['n']} players whose matches are over.")
        if week.get("inside") is not None:
            cols[3].metric("Inside their range", f"{week['inside']:.0%}",
                           help=f"Of those scores, inside the middle 80% of simulated scores (about 80% is right). "
                                f"Hauls (10+): {week['hauls']} vs {week['hauls_expected']:.1f} expected.")
    model_team = data["teams"].get("model")
    if model_team and model_team.get("made_at") and data.get("deadline"):
        gap = pd.Timestamp(data["deadline"]) - pd.Timestamp(model_team["made_at"])
        if gap > pd.Timedelta(days=2):
            st.warning(f"This Model's Team was picked {gap.days} days before the deadline and not again since, so it missed "
                       "the team news in between. Run `xpfpl recommend` (or Plan Ahead) on deadline day.")

    teams = {label: data["teams"][k] for k, label in (("model", "This Model's Team"), ("mine", "My Team")) if data["teams"].get(k)}
    if teams:
        st.markdown("#### The teams, live")
        which = st.segmented_control("Team", list(teams), default=list(teams)[0], key="live_team") or list(teams)[0]
        team = teams[which]
        picks = pd.DataFrame(team["picks"]).set_index("element")
        picks["Player"] = picks.index.map(name)
        picks["Club"] = players["team"].reindex(picks.index).map(team_short)
        picks["Counted"] = picks["points"] * picks["multiplier"]
        picks = picks.join(t.reindex(columns=["Pos", "pts_p10", "pts_p90", "error", "verdict", "Why"]), how="left")
        settled = picks["state"] == live.DONE
        picks["Range"] = [f"{a:.0f}-{b:.0f}" if a == a and b == b else "" for a, b in zip(picks["pts_p10"], picks["pts_p90"])]
        picks["Diff"] = picks["error"].where(settled)
        picks["Mins"] = [f"{m:.0f}" if s == live.DONE or m else STATES.get(s, s) for m, s in zip(picks["minutes"], picks["state"])]
        st.dataframe(picks[["Player", "Club", "Pos", "role", "xp", "Range", "Counted", "Mins", "Diff", "verdict", "Why"]]
                     .rename(columns={"role": "Picked", "xp": "xP", "Counted": "Points", "verdict": "Mainly"}),
                     hide_index=True, width="stretch",
                     column_config={"xP": st.column_config.NumberColumn(format=style.POINTS),
                                    "Diff": st.column_config.NumberColumn(format="%+.1f", help="His points minus xP, once his matches are over")})
        notes = [f"{name(s['in'])} on for {name(s['out'])}" for s in team["subs"]]
        if team["armband"] not in (None, team["captain"]):
            notes.append(f"armband to {name(team['armband'])} (the captain didn't play)")
        if notes:
            st.caption("Automatic substitutions: " + "; ".join(notes) + ".")

    if week.get("n"):
        st.markdown("#### Where the misses came from")
        rows = [_parts(week, "All")] + [_parts(s, pos) for pos, s in week.get("positions", {}).items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format=style.POINTS)
                                    for c in ["xP", "Points", "Points − xP", *PARTS.values()]})
        st.caption("Points minus xP per player expected to feature, split into three parts that add up to it. "
                   "**Minutes**: played when he might not have (+), or didn't (−), against the chance he was given. "
                   "**Finishing**: goals and assists beyond or short of the match's xG and xA (mostly luck). "
                   "**Chances & the rest**: how many chances he got against his xP, clean sheets, bonus, saves, "
                   "defensive contribution, cards. A minutes or rest part that stays on one side week after week is "
                   "what to fix in the model; finishing evens out.")

    # Everyone else: the players in the teams are in their own table.
    in_teams = {p["element"] for team in data["teams"].values() for p in team["picks"]}
    done = t[(t["state"] == live.DONE) & ~t.index.isin(list(in_teams))].copy()
    if len(done):
        st.markdown("#### Who beat the forecast, and who fell short")
        if in_teams:
            st.caption("Everyone else: the players in the teams above are in their table.")
        side = st.segmented_control("Show", ["Fell short", "Beat it", "Biggest either way"], default="Fell short", key="live_side") or "Fell short"
        shown = done[done["error"] < 0] if side == "Fell short" else done[done["error"] > 0] if side == "Beat it" else done
        shown = shown.reindex(shown["error"].abs().sort_values(ascending=False).index).head(30)
        shown = shown.reindex(columns=[*shown.columns, *{"pts_p10", "pts_p90"} - set(shown.columns)])
        shown["Range"] = [f"{a:.0f}-{b:.0f}" if a == a and b == b else "" for a, b in zip(shown["pts_p10"], shown["pts_p90"])]
        shown = shown.rename(columns={"xp": "xP", "points": "Points", "minutes": "Mins", "error": "Diff", "verdict": "Mainly",
                                      "minutes_part": "Minutes", "finishing": "Finishing", "other": "Chances & the rest"})
        st.dataframe(shown[["Player", "Club", "Pos", "xP", "Range", "Points", "Mins", "Diff", "Mainly", "Minutes", "Finishing",
                            "Chances & the rest", "Why"]], hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format=style.POINTS)
                                    for c in ("xP", "Diff", "Minutes", "Finishing", "Chances & the rest")})

    st.markdown("#### Before the next deadline")
    any_lesson = False
    for key, (title, text) in LESSONS.items():
        ids = data["lessons"].get(key) or []
        if ids:
            any_lesson = True
            st.markdown(f"**{title}**: {', '.join(name(e) for e in ids)}")
            st.caption(text)
    if not any_lesson:
        st.caption("Nothing stands out yet.")

    report = scorecard.load(api.current_season(bs))
    diag = pd.DataFrame([{k: v for k, v in d.items() if k != "positions"} for d in (report or {}).get("diagnosis", [])
                         if d["model"] == data["model"]])
    if len(diag):
        st.markdown("#### Earlier gameweeks")
        st.dataframe(diag[["gw", "n", "error", *PARTS]].rename(columns={"gw": "GW", "n": "Players", "error": "Points − xP", **PARTS}),
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format=style.POINTS) for c in ["Points − xP", *PARTS.values()]})
        st.caption("The same split for every gameweek scored so far (`xpfpl scorecard`), per player getting minutes.")
