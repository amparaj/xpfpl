"""The dashboard's Team News tab: who's out, doubtful or back and when, your squad's news first,
the managers' press conferences and every change to FPL's news this season (data/news.py).

Today's FPL news comes from the dashboard's own bootstrap-static call (as fresh as the last
"Refresh live FPL data"); the change log comes from the hourly Action's news.json on the `news`
branch (or archive/news/ when offline); the press conferences are read from the web at most every
30 minutes.
"""

import pandas as pd
import streamlit as st

from xpfpl import config
from xpfpl.data import api, news

STATUS_ORDER = ["Injured", "Suspended", "Doubtful", "Not available", "Unavailable", "Available"]
RULES = {"flag": "FPL's flag", "press": "Press conference", "back": "Return date", "risk": "Known absence"}
ELSEWHERE = [
    ("Fantasy Football Scout: injuries and bans", "https://www.fantasyfootballscout.co.uk/fantasy-football-injuries"),
    ("NewsNow: Premier League injuries and suspensions",
     "https://www.newsnow.com/au/Sport/Football/Premier+League/Injuries+and+Suspensions"),
    ("Premier Fantasy Tools: press conference summaries", news.PRESS_URL),
]


@st.cache_data(ttl=600, show_spinner="Loading the news log...")
def tracked() -> dict | None:
    return news.remote()


@st.cache_data(ttl=1800, show_spinner="Reading the press conferences...")
def press(_bs: dict, gw_next: int | None) -> dict | None:
    """Today's press conferences (cached half an hour; `gw_next` keys the cache)."""
    return news.press_now(_bs)


def _ago(t: pd.Timestamp, now: pd.Timestamp) -> str:
    if pd.isna(t):
        return ""
    hours = (now - t).total_seconds() / 3600
    if hours < 1:
        return "under an hour ago"
    if hours < 48:
        return f"{hours:.0f} h ago"
    return f"{hours / 24:.0f} days ago"


def _table(state: dict, team_short: dict, xp: pd.Series | None, avail: pd.DataFrame | None) -> pd.DataFrame:
    t = pd.DataFrame(state["players"])
    if t.empty:
        return t
    now = pd.Timestamp(state["updated"])
    added = pd.to_datetime(t["added"], utc=True, errors="coerce")
    t = t.assign(
        Status=t["status"].map(news.STATUS_WORDS), Team=t["team"].map(team_short), Pos=t["pos"].map(config.POSITIONS),
        Back=pd.to_datetime(t["back"], errors="coerce").dt.strftime("%a %d %b").fillna(""),
        Updated=added.dt.tz_convert("Europe/London").dt.strftime("%a %d %b %H:%M").fillna(""),
        Age=[_ago(a, now) for a in added], age_days=(now - added).dt.total_seconds() / 86400,
        Press=t["press"].fillna(""),
        Ahead=["; ".join(f"GW{r['gw']}: {r['notes']}" for r in risks) if isinstance(risks, list) else ""
               for risks in t["risks"]],
    )
    if xp is not None:
        t["xp"] = t["id"].map(xp)
    if avail is not None and "avail" in avail:
        t["used"] = t["id"].map(avail["avail"]) * 100
        t["rule"] = t["id"].map(avail["news_rule"]).map(RULES).fillna("")
    return t


COLUMNS = {"name": "Player", "Team": "Team", "Pos": "Pos", "Status": "Status", "chance": "Chance %",
           "reason": "Reason", "Back": "Expected back", "news": "FPL news", "Updated": "Updated (UK)", "Age": "Age",
           "Press": "Press conference", "Ahead": "Known absences", "used": "Chance used %", "rule": "Set by",
           "xp": "xP next GW", "selected": "Sel %", "source": "Source"}


def _config() -> dict:
    return {
        "Chance %": st.column_config.NumberColumn(format="%d", help="FPL's chance of playing next round"),
        "Chance used %": st.column_config.NumberColumn(
            format="%d", help="The chance of playing this model used for the next gameweek (data/news.py)"),
        "Set by": st.column_config.TextColumn(help="What set that chance: FPL's flag, a newer press conference, ..."),
        "Source": st.column_config.LinkColumn(display_text="Source", help="The article FPL's news came from"),
        "xP next GW": st.column_config.NumberColumn(format="%.2f"),
        "Sel %": st.column_config.NumberColumn(format="%.1f"),
        "Updated (UK)": st.column_config.TextColumn(help="When FPL last changed this player's news (news_added)"),
        "FPL news": st.column_config.TextColumn(width="large"),
    }


def _shown(t: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return t[[c for c in cols if c in t]].rename(columns=COLUMNS)


def render(bs: dict, team_short: dict, me=None, xp: pd.Series | None = None,
           avail: pd.DataFrame | None = None) -> None:
    """`me`: myteam.MyTeam (or None); `xp`: next gameweek's xP by element; `avail`: the forecast's
    `avail` and `news_rule` columns by element (or None for an older forecast)."""
    remote = tracked()
    season = api.current_season(bs)
    previous = remote if remote and remote.get("season") == season else None
    gw_next = next((ev["id"] for ev in bs["events"] if ev["is_next"]), None)
    state = news.update(previous, bs, press=press(bs, gw_next), read_press=False)
    t = _table(state, team_short, xp, avail)
    now = pd.Timestamp(state["updated"])
    p = state.get("press") or {}

    out = t[t["status"].isin(["i", "s", "n"])] if len(t) else t
    c = st.columns(4)
    c[0].metric("Injured or suspended", len(out))
    c[1].metric("Doubtful", int((t["status"] == "d").sum()) if len(t) else 0)
    c[2].metric("News changed in the last 48 h", int((t["age_days"] <= 2).sum()) if len(t) else 0)
    c[3].metric("Press conferences", f"GW{gw_next}" if p.get("fresh") else "not yet",
                help=f"{news.PRESS_NAME}, last updated {p.get('updated') or 'never'}")
    st.caption("FPL's own news for every player (status, chance of playing, reason, return date and the article it "
               "came from), as fresh as the last 'Refresh live FPL data'; the press conferences from "
               f"[{news.PRESS_NAME}]({news.PRESS_URL}). For the next gameweek this model uses FPL's chance of playing "
               "unless a press conference given after FPL's last update says otherwise (OUT 0, DOUBT "
               f"{config.PRESS_DOUBT_CHANCE:.0%}, IN 100%); for the weeks after, FPL's return dates (0 before, "
               f"{config.RETURN_CHANCE:.0%} from the date, 100% after a ban) and known absences.")

    cols = ["name", "Team", "Pos", "Status", "chance", "used", "rule", "reason", "Back", "Press", "Updated", "Age",
            "xp", "selected", "source", "news", "Ahead"]
    if me is not None and len(t):
        st.markdown("**Your squad**")
        mine = t[t["id"].isin(list(me.squad))]
        if mine.empty:
            st.success("No news for any of your 15.")
        else:
            st.dataframe(_shown(mine.sort_values("chance", na_position="last"), cols), hide_index=True,
                         width="stretch", column_config=_config())

    st.markdown("**All players with news**")
    f = st.columns([2, 2, 1, 2])
    status = f[0].multiselect("Status", STATUS_ORDER, default=["Injured", "Suspended", "Doubtful"], key="news_status")
    clubs = f[1].multiselect("Club", sorted(team_short.values()), key="news_clubs")
    recent = f[2].selectbox("Changed", ["Any time", "Last 48 h", "Last 7 days"], key="news_recent")
    search = f[3].text_input("Search", key="news_search")
    shown = t
    if len(shown):
        if status:
            shown = shown[shown["Status"].isin(status)]
        if clubs:
            shown = shown[shown["Team"].isin(clubs)]
        if recent != "Any time":
            shown = shown[shown["age_days"] <= (2 if recent == "Last 48 h" else 7)]
        if search:
            shown = shown[shown["name"].str.contains(search, case=False, regex=False)]
        st.dataframe(_shown(shown.sort_values(["age_days", "name"], na_position="last"), cols), hide_index=True,
                     width="stretch", height=480, column_config=_config())
        st.caption(f"{len(shown)} players, latest news first. 'Expected back' is FPL's date: he's expected to be "
                   "available for matches from that day. Players who've left the club are under 'Unavailable'.")

    st.markdown(f"**Press conferences** · {news.PRESS_NAME}" + (f", last updated {p['updated']}" if p.get("updated") else ""))
    if not p.get("clubs"):
        st.caption("Couldn't read the press-conference page.")
    else:
        if not p.get("fresh"):
            st.warning(f"These are from {p.get('updated')}, before the last deadline: not yet updated for GW{gw_next}, "
                       "so this model isn't using them.")
        cols_ = st.columns(2)
        for i, club in enumerate(p["clubs"]):
            with cols_[i % 2].expander(club["club"], expanded=False):
                for said in news.STATUSES:
                    names = [x for x in club["players"] if x["status"] == said]
                    if names:
                        st.markdown(f"**{said}**: " + ", ".join(
                            x["name"] + ("" if x.get("id") is not None else " (not matched to an FPL player)")
                            for x in names))
                for q in club["quotes"]:
                    st.markdown(f"> {q['text']}" + (f"\n>\n> — {q['by']}" if q.get("by") else ""))
        st.caption(f"Summaries by [{news.PRESS_NAME}]({news.PRESS_URL}); read {p.get('fetched', '')[:16].replace('T', ' ')} UTC.")

    st.markdown("**Changes to FPL's news**")
    log = news.change_log(state)
    if log.empty and state.get("season"):
        log = news.archived_log(state["season"])
    if previous is None:
        st.caption("The hourly news log isn't available (the Action hasn't run yet, or you're offline): this is "
                   "everyone's latest news, without the history.")
    log = log[log["t"] >= now - pd.Timedelta(days=14)] if len(log) else log
    if len(log):
        code_team = {t_["code"]: t_["id"] for t_ in bs["teams"]}
        st.dataframe(log.assign(
            when=log["t"].dt.tz_convert("Europe/London").dt.strftime("%a %d %b %H:%M"),
            Team=log["team_code"].map(code_team).map(team_short), Pos=log["pos"].map(config.POSITIONS),
            Status=log["status"].map(news.STATUS_WORDS), news=log["news"].replace("", "Back to full fitness"))[
            ["when", "name", "Team", "Pos", "Status", "chance", "news", "source"]].rename(columns={
                "when": "When (UK)", "name": "Player", "chance": "Chance %", "news": "FPL news", "source": "Source"}),
            hide_index=True, width="stretch", height=320,
            column_config={"Source": st.column_config.LinkColumn(display_text="Source"),
                           "Chance %": st.column_config.NumberColumn(format="%d")})
        st.caption("The last 14 days, timed by FPL's own update time.")

    st.markdown("**Elsewhere**")
    st.markdown("\n".join(f"- [{label}]({url})" for label, url in ELSEWHERE))
    st.caption("Fantasy Football Scout's table adds a return date and a news source for some players, but its terms "
               "don't allow copying it automatically, so it's linked rather than read.")
