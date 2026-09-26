"""Streamlit dashboard for xP-FPL. Launch with `xpfpl app` (or `streamlit run src/xpfpl/app.py`)."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from html import escape

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from xpfpl import chips, config, guide, market_view, models, review
from xpfpl.data import api, cups
from xpfpl.data.history import load_matches
from xpfpl.myteam import load_my_team
from xpfpl.optimise import solve
from xpfpl.predict import predict_upcoming
from xpfpl.style import CLUB_COLOURS, CLUB_TEXT, DECIMAL, POINTS

SETTINGS_PATH = config.DATA_DIR / "app_settings.json"  # data/ is gitignored, so your team id stays local
SERIES = "#2a78d6"   # one hue for single-series charts
MUTED = "#8a8984"    # reference lines (GW average)
FORECAST = "#d9822b"  # the model's forecast, next to the points actually scored
FDR_BG = {1: "#cde2fb", 2: "#9ec5f4", 3: "#6da7ec", 4: "#256abf", 5: "#104281"}  # sequential blue
FDR_FG = {1: "#0b0b0b", 2: "#0b0b0b", 3: "#0b0b0b", 4: "#ffffff", 5: "#ffffff"}
PHOTO_URL = "https://resources.premierleague.com/premierleague25/photos/players/110x140/{code}.png"
SHIRT_URL = "https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{team_code}{gk}-66.png"
PITCH_CSS = """<style>
.xp-pitch { max-width: 860px; margin: 0 auto; padding: 12px 8px 4px; border-radius: 10px 10px 0 0;
  background: repeating-linear-gradient(180deg, #2e8b47 0 58px, #29803f 58px 116px);
  box-shadow: inset 0 0 0 2px rgba(255,255,255,.35); }
.xp-bench { max-width: 860px; margin: 0 auto; padding: 6px 8px 8px; border-radius: 0 0 10px 10px;
  background: #dfe6e0; }
.xp-row { display: flex; justify-content: center; gap: 12px; margin: 6px 0 10px; }
.xp-card { position: relative; width: 110px; text-align: center; font-size: 12.5px; line-height: 1.4; }
.xp-face { width: 64px; height: 64px; margin: 0 auto 4px; border-radius: 50%; border: 3px solid;
  background: #f4f4f2 center 2px / 100% auto no-repeat; position: relative; }
.xp-shirt { background-size: 80% auto; background-position: center 55%; }
.xp-crest { position: absolute; width: 26px; right: -10px; bottom: -4px; }
.xp-name { font-weight: 700; font-size: 13.5px; padding: 1px 4px; border-radius: 4px 4px 0 0;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.xp-fix { padding: 0 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.xp-mid { padding: 0 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; background: #fff4d6;
  color: #5c4400; font-size: 11px; }
.xp-pts { background: #ffffff; color: #111111; padding: 0 4px; border-radius: 0 0 4px 4px; }
.xp-badge { position: absolute; top: 0; z-index: 1; min-width: 21px; height: 21px; padding: 0 4px;
  border-radius: 11px; font-size: 11.5px; font-weight: 700; line-height: 21px; }
.xp-cap { right: 8px; background: #111111; color: #ffffff; }
.xp-vc { right: 8px; background: #ffffff; color: #111111; border: 1px solid #111111; line-height: 19px; }
.xp-flag { left: 8px; color: #111111; cursor: help; }
.xp-order { font-size: 11.5px; font-weight: 700; color: #3d4a40; }
</style>"""
ACTIVE_CHIPS = {"None": None, "Wildcard": "wildcard", "Free Hit": "freehit",
                "Triple Captain": "3xc", "Bench Boost": "bboost"}

st.set_page_config(page_title="xP-FPL", page_icon="⚽", layout="wide")
st.markdown("""
<style>
[data-testid="stDataFrame"] [role="gridcell"],
[data-testid="stDataFrame"] [role="columnheader"] {
    font-size: 0.82rem;
    padding: 0.2rem 0.35rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- settings and cached data

def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(**kwargs) -> None:
    settings = {**load_settings(), **kwargs}
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def stamp() -> tuple[float, ...]:
    """Changes whenever the data or any trained model changes, so cached results refresh."""
    paths = [config.MATCHES_PATH] + [models.path(n) for n in models.NAMES if n != "baseline"]
    return tuple(p.stat().st_mtime if p.exists() else 0.0 for p in paths)


@st.cache_data(ttl=600, show_spinner="Loading FPL data...")
def live_data_at() -> tuple[dict, list[dict], datetime]:
    """bootstrap-static, the fixture list, and when they were fetched."""
    return api.bootstrap(), api.fixtures(), datetime.now()


def live_data() -> tuple[dict, list[dict]]:
    bs, fx, _ = live_data_at()
    return bs, fx


@st.cache_data(show_spinner="Predicting xP...")
def predictions(horizon: int, model: str, data_stamp) -> tuple[pd.DataFrame, list[int]]:
    bs, fx = live_data()
    return predict_upcoming(horizon, model, bs, fx)


@st.cache_data(ttl=300, show_spinner="Loading your team...")
def my_team(team_id: int):
    bs, _ = live_data()
    return load_my_team(team_id, bs)


@st.cache_data(ttl=300, show_spinner="Loading your history...")
def manager(team_id: int) -> tuple[dict, dict, list[dict]]:
    return api.entry(team_id), api.entry_history(team_id), api.entry_transfers(team_id)


@st.cache_data(show_spinner="Optimising...")
def run_solve(horizon: int, model: str, gameweeks: tuple[int, ...], data_stamp, **kwargs):
    players, _ = predictions(horizon, model, data_stamp)
    return solve(players, list(gameweeks), **kwargs)


@st.cache_data(show_spinner="Checking chips...")
def run_chip_advice(horizon: int, model: str, data_stamp, team_id: int, available: tuple[str, ...], **kwargs):
    players, gameweeks = predictions(horizon, model, data_stamp)
    bs, _ = live_data()
    # The chips are judged against the same plan the page is showing, week-by-week planning included.
    plan = solve(players, gameweeks, **kwargs)
    windows = {c["name"]: c for c in bs["chips"] if c["name"] in available}
    base = {k: kwargs[k] for k in ("current_squad", "bank", "must_have", "banned")}
    return chips.advise(players, gameweeks, plan, windows, base)


@st.cache_data(ttl=300, show_spinner="Loading gameweek...")
def gw_review(team_id: int, gw: int):
    bs, _ = live_data()
    points = review.gameweek_points(gw)
    r = review.review_picks(team_id, gw, bs, points)
    best, plan = review.hindsight_best(r["picks"], gw, r["chip"])
    return points, r, best, plan


@st.cache_data(show_spinner="Loading the model's forecasts for the season...")
def season_xp(season: str, model: str, data_stamp) -> pd.DataFrame:
    """gw, element, xp, source for every played gameweek: the forecast saved before each deadline,
    else an in-sample rebuild (review.season_forecasts)."""
    return review.season_forecasts(load_matches(), season, model)


def gw_xp(season: str, gw: int, model: str) -> tuple[pd.Series, str | None]:
    """One gameweek's xP per player, and where it came from (review.SAVED / review.REBUILT)."""
    week = season_xp(season, model, stamp())
    week = week[week["gw"] == gw]
    return week.set_index("element")["xp"], (week["source"].iloc[0] if len(week) else None)


@st.cache_data(show_spinner="Loading your season's picks...")
def team_forecasts(team_id: int, gws: tuple[int, ...], season: str, model: str, data_stamp) -> pd.DataFrame:
    """event, forecast, source: what the model expected your team to score each gameweek, as picked."""
    xp = season_xp(season, model, data_stamp)
    rows = []
    for gw in gws:
        week = xp[xp["gw"] == gw]
        if week.empty:
            continue
        picks = pd.DataFrame(api.entry_picks(team_id, gw)["picks"]).set_index("element")
        rows.append({"event": gw, "forecast": review.team_forecast(picks, week.set_index("element")["xp"]),
                     "source": week["source"].iloc[0]})
    return pd.DataFrame(rows, columns=["event", "forecast", "source"])


def xp_source_note(sources) -> str:
    """How to read past xP, given which sources were used."""
    sources = set(sources)
    if sources <= {review.SAVED}:
        return "xP is the forecast saved before each deadline, so it's exactly what the model said at the time."
    if review.SAVED in sources:
        return ("xP is the forecast saved before each deadline where there is one; the other weeks are rebuilt "
                "by today's model, which was also trained on them, so those are a little optimistic.")
    return ("No forecast was saved before this deadline, so xP is rebuilt by today's model, which was also "
            "trained on this season: a little optimistic.")


@st.cache_data(ttl=600)
def midweek_badges(season: str, gameweeks: tuple[int, ...]) -> dict[tuple[int, int], str]:
    """(team id, gw) -> "UCL Tue": each club's cup or European match before those gameweeks."""
    team_id = {t["code"]: t["id"] for t in live_data()[0]["teams"]}
    return {(team_id[code], gw): label for (code, gw), label in cups.badges(season, gameweeks).items()
            if code in team_id}


@st.cache_data(ttl=600)
def midweek_results(season: str, gw: int) -> pd.DataFrame:
    """The cup and European matches played before `gw`, one row each."""
    m = cups.matches(season)
    return m[(m["gw"] == gw) & m["finished"]]


def fixture_labels(bs: dict, fx: list[dict], gameweeks: list[int]) -> dict[tuple[int, int], str]:
    """(team id, gw) -> "ARS (H)" style label; blanks are missing and doubles are joined."""
    short = {t["id"]: t["short_name"] for t in bs["teams"]}
    out: dict[tuple[int, int], list[str]] = {}
    for f in fx:
        if f["event"] in gameweeks:
            out.setdefault((f["team_h"], f["event"]), []).append(f"{short[f['team_a']]} (H)")
            out.setdefault((f["team_a"], f["event"]), []).append(f"{short[f['team_h']]} (A)")
    return {k: ", ".join(v) for k, v in out.items()}


def fixture_difficulty(fx: list[dict], gameweeks: list[int]) -> dict[tuple[int, int], int]:
    """(team id, gw) -> FPL difficulty 1-5; a double gameweek takes the rounded mean of its fixtures."""
    out: dict[tuple[int, int], list[int]] = {}
    for f in fx:
        if f["event"] in gameweeks:
            out.setdefault((f["team_h"], f["event"]), []).append(f["team_h_difficulty"])
            out.setdefault((f["team_a"], f["event"]), []).append(f["team_a_difficulty"])
    return {k: round(sum(v) / len(v)) for k, v in out.items()}


def run_cli(command: str) -> bool:
    """Run `xpfpl <command>` in a subprocess and stream its output into the page."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}
    with st.status(f"Running `xpfpl {command}`...", expanded=True) as status:
        box, lines = st.empty(), []
        proc = subprocess.Popen([sys.executable, "-m", "xpfpl", command], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
        for line in proc.stdout:
            lines.append(line.rstrip())
            box.code("\n".join(lines[-25:]))
        ok = proc.wait() == 0
        status.update(label=f"`xpfpl {command}` {'finished' if ok else 'failed'}",
                      state="complete" if ok else "error", expanded=not ok)
    return ok


# ---------------------------------------------------------------- sidebar

bs, fx = live_data()
next_gw = api.next_gameweek(bs)
last_gw = api.last_finished_gameweek(bs)
season = api.current_season(bs)
events = {e["id"]: e for e in bs["events"]}
elements = pd.DataFrame(bs["elements"]).set_index("id")
team_short = {t["id"]: t["short_name"] for t in bs["teams"]}
team_colours = {t["id"]: CLUB_COLOURS.get(t["short_name"], "#d9dde3") for t in bs["teams"]}
team_text_colours = {t["id"]: CLUB_TEXT.get(t["short_name"], "#111111") for t in bs["teams"]}
team_colours_short = {team_short[team_id]: colour for team_id, colour in team_colours.items()}
team_text_colours_short = {team_short[team_id]: colour for team_id, colour in team_text_colours.items()}
player_label = {pid: f"{r.web_name} ({team_short[r.team]}, {config.POSITIONS[r.element_type]}, £{r.now_cost / 10:.1f})"
                for pid, r in elements.iterrows()}


def style_team_column(frame: pd.DataFrame, team_ids: pd.Series):
    """Add a readable club-colour accent to a displayed Team column."""
    def style(column: pd.Series) -> list[str]:
        if column.name != "Team":
            return [""] * len(column)
        return [f"background-color: {team_colours.get(team_id, '#d9dde3')}; "
                f"color: {team_text_colours.get(team_id, '#111111')}" for team_id in team_ids]
    return frame.style.apply(style, axis=0)


def countdown(delta) -> str:
    seconds = int(delta.total_seconds())
    days, seconds = divmod(seconds, 24 * 60 * 60)
    hours, seconds = divmod(seconds, 60 * 60)
    minutes, seconds = divmod(seconds, 60)
    clock = f"{hours}h {minutes}m {seconds:02d}s"
    return f"{days}d {clock}" if days else clock


@st.fragment(run_every="1s")
def gameweek_panel(data_gw: int) -> None:
    """The current gameweek's progress and the next deadline, redrawn every second (live FPL data is
    cached for 10 minutes, so match progress moves at that pace)."""
    bs_now, fx_now = live_data()
    gw_next = api.next_gameweek(bs_now)
    event = next(e for e in bs_now["events"] if e["id"] == gw_next)
    deadline = datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00")).astimezone()
    left = deadline - datetime.now().astimezone()
    st.metric(f"GW{gw_next} deadline · {deadline:%a %d %b %H:%M}",
              countdown(left) if left.total_seconds() > 0 else "passed")
    if left.total_seconds() <= 0:
        st.caption("Click **Refresh live FPL data** to move on to the next gameweek.")

    status = api.gameweek_status(bs_now, fx_now)
    gw = status["gw"]
    if status["phase"] == "pre-season":
        st.info(f"Season {season} hasn't started yet.")
    elif status["phase"] == "live":
        st.info(f"**GW{gw} in progress**: {status['played']} of {status['matches']} matches played.")
        if status["last_kickoff"]:
            st.caption(f"Last match kicks off {status['last_kickoff'].astimezone():%a %d %b %H:%M}. The gameweek "
                       "finishes when its bonus points are added (~1-2 h after the final whistle), then FPL "
                       "confirms the data a few hours later.")
    elif status["phase"] == "checking":
        st.warning(f"**GW{gw} finished**, FPL is still confirming the data (bonus points, corrections), "
                   "usually for a few hours. Wait for it before fetching: fetched match data is cached.")
    elif data_gw >= gw:
        st.success(f"**GW{gw} complete** and fetched.")
    else:
        st.success(f"**GW{gw} complete**: ready to fetch (steps 1-3 below).")


with st.sidebar:
    st.title("xP-FPL")
    st.caption(f"Season {season}")
    settings = load_settings()
    team_id = st.number_input("FPL team id", min_value=0, step=1, value=int(settings.get("team_id", 0)),
                              help="The number in the URL of your FPL Points page.")
    if team_id and team_id != settings.get("team_id"):
        save_settings(team_id=int(team_id))

    data_gw = 0
    if config.MATCHES_PATH.exists():
        m = pd.read_parquet(config.MATCHES_PATH, columns=["season", "gw"])
        data_gw = int(m.loc[m["season"] == season, "gw"].max() or 0)

    gameweek_panel(data_gw)

    st.markdown("**Any time before a Gameweek deadline:**")
    if st.button("Refresh live FPL data", width="stretch",
                 help="Re-downloads today's prices, injury flags, fixtures and your team from the FPL API "
                      "(otherwise cached for 5-10 min). Nothing is saved and nothing needs retraining."):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Live FPL data last refreshed: **{live_data_at()[2]:%a %d %b %H:%M}**")

    st.divider()
    st.subheader("Data and model")
    default_model_path = models.path(config.MODEL)
    model_time = (datetime.fromtimestamp(default_model_path.stat().st_mtime)
                  if default_model_path.exists() else None)

    # The order to run them in: results feed the odds (matched to fixtures), and both feed training.
    st.markdown("**For the start of the Season/After a Gameweek finishes:**")
    st.write(f"Match data up to **GW{data_gw}** (last finished: GW{last_gw})")
    confirmed_gw = max((e["id"] for e in bs["events"] if e["data_checked"]), default=0)
    if data_gw < confirmed_gw:
        st.warning(f"GW{confirmed_gw} is complete. Run steps 1-3 below.")
    elif data_gw < last_gw:
        st.info(f"GW{last_gw} has finished, but FPL is still confirming its data. Fetch once it's confirmed.")
    elif model_time and config.MATCHES_PATH.stat().st_mtime > default_model_path.stat().st_mtime:
        st.warning("The data is newer than the model. Run step 3 (Retrain) to use it.")
    if st.button("1. Fetch match data", width="stretch",
                 help="xpfpl fetch (~1 min): rebuilds the training dataset (every player's stats in every "
                      "finished match, history + this season) and saves it to disk"):
        if run_cli("fetch"):
            st.cache_data.clear()
    if st.button("2. Fetch betting odds", width="stretch",
                 help="xpfpl markets: Polymarket odds for every match since 2024-25, "
                      "saved to disk (a few minutes the first time)"):
        if run_cli("markets"):
            st.cache_data.clear()
    from xpfpl.data import markets as market_data
    odds_time = (datetime.fromtimestamp(market_data.MATCHES_PATH.stat().st_mtime)
                 if market_data.MATCHES_PATH.exists() else None)
    st.caption(f"Odds last fetched: **{odds_time:%a %d %b %H:%M}**" if odds_time
               else "Odds not fetched yet.")
    if st.button("3. Retrain", width="stretch",
                 help="xpfpl train (~30 s): refits the model on the match data and odds above"):
        if run_cli("train"):
            st.cache_data.clear()
    st.caption(f"Model (`{config.MODEL}`) trained: **{model_time:%a %d %b %H:%M}**"
               if model_time else f"Model (`{config.MODEL}`): **not trained**")
    trained_models = models.trained()
    if len(trained_models) > 1:
        st.caption("Also trained: " + ", ".join(n for n in trained_models
                                                if n not in (config.MODEL, "baseline")) or "")
    if st.button("4. Publish the website", width="stretch",
                 help="xpfpl publish (~1 min): exports the season, your team, the odds and the accuracy "
                      "reports, builds the site in web/ and pushes it to the gh-pages branch (GitHub Pages)"):
        run_cli("publish")
    meta_path = config.SITE_DATA_DIR / "meta.json"
    st.caption(f"Website data exported: **{datetime.fromtimestamp(meta_path.stat().st_mtime):%a %d %b %H:%M}**"
               if meta_path.exists() else "Website not published yet.")

if not team_id:
    st.info("Enter your FPL team id in the sidebar to get started.")
    st.stop()
if not config.MATCHES_PATH.exists():
    st.warning("No match data yet. Run the three steps in the sidebar: **Fetch match data**, "
               "**Fetch betting odds**, then **Retrain**.")
    st.stop()

# The Guide's weekly routine, left to right: look back at the week, research, plan; then the season.
tab_guide, tab_review, tab_players, tab_markets, tab_plan, tab_season = st.tabs(
    ["Guide", "Gameweek Review", "Players & Fixtures", "Markets", "Plan Ahead", "My Season"])

with tab_guide:
    guide.render()


# ---------------------------------------------------------------- plan ahead

@st.cache_data(ttl=86400, show_spinner=False)
def photo_urls(codes: tuple[int, ...]) -> dict[int, str | None]:
    """Premier League headshot per player code, or None where the CDN has no photo (~1 in 5)."""
    def check(code: int) -> str | None:
        url = PHOTO_URL.format(code=code)
        try:
            return url if requests.head(url, timeout=5).status_code == 200 else None
        except requests.RequestException:
            return None
    with ThreadPoolExecutor(8) as pool:
        return dict(zip(codes, pool.map(check, codes)))


def show_pitch(players: pd.DataFrame, plan, gw: int, labels: dict, fdr: dict, midweek: dict | None = None) -> None:
    """The XI laid out by position on a pitch, then the bench in auto-sub order, as one HTML block.
    `midweek`: (team id, gw) -> "UCL Tue", a line under the fixture for a club that plays midweek."""
    midweek = midweek or {}
    col = f"xp_{gw}"
    squad = plan.lineups[gw] + plan.bench[gw]
    photos = photo_urls(tuple(sorted(int(elements.at[p, "code"]) for p in squad)))

    def card(p: int, note: str = "") -> str:
        r, e = players.loc[p], elements.loc[p]
        shirt = SHIRT_URL.format(team_code=e["team_code"], gk="_1" if r["position"] == 1 else "")
        photo = photos.get(int(e["code"]))
        club, club_text = team_colours.get(r["team"], "#d9dde3"), team_text_colours.get(r["team"], "#111111")
        level = fdr.get((r["team"], gw))
        fix_bg, fix_fg = (FDR_BG[level], FDR_FG[level]) if level else ("#f0efec", "#555555")
        badges = ""
        if p == plan.captains[gw]:
            badges += "<span class='xp-badge xp-cap'>C</span>"
        elif p == plan.vice_captains[gw]:
            badges += "<span class='xp-badge xp-vc'>V</span>"
        chance = r["chance"]
        if pd.notna(chance) and chance < 100:
            badges += (f"<span class='xp-badge xp-flag' style='background:{'#d0021b' if chance <= 25 else '#f5a623'}' "
                       f"title='{escape(str(e['news']), quote=True)}'>{int(chance)}</span>")
        face = (f"<div class='xp-face' style='border-color:{club};background-image:url({photo})'>"
                f"<img class='xp-crest' src='{shirt}' alt=''></div>" if photo else
                f"<div class='xp-face xp-shirt' style='border-color:{club};background-image:url({shirt})'></div>")
        return (f"<div class='xp-card' title='{escape(e['first_name'])} {escape(e['second_name'])} · "
                f"{r['team_name']} · £{r['price']:.1f}m'>{note}{badges}{face}"
                f"<div class='xp-name' style='background:{club};color:{club_text}'>{escape(r['name'])}</div>"
                f"<div class='xp-fix' style='background:{fix_bg};color:{fix_fg}'>"
                f"{labels.get((r['team'], gw), 'no fixture')}</div>"
                + (f"<div class='xp-mid'>{midweek[(r['team'], gw)]} midweek</div>" if (r["team"], gw) in midweek else "")
                + f"<div class='xp-pts'><b>{r[col]:.2f}</b> xP · £{r['price']:.1f}</div></div>")

    rows = "".join(
        "<div class='xp-row'>" + "".join(card(p) for p in plan.lineups[gw] if players.at[p, "position"] == pos)
        + "</div>" for pos in (1, 2, 3, 4))
    bench = "".join(card(p, f"<div class='xp-order'>{'GKP' if i == 0 else i}</div>")
                    for i, p in enumerate(plan.bench[gw]))
    st.html(PITCH_CSS + f"<div class='xp-pitch'>{rows}</div>"
            f"<div class='xp-bench'><div class='xp-row'>{bench}</div></div>")


with tab_plan:
    me = my_team(int(team_id))
    with st.form("plan"):
        c = st.columns([1, 1, 1.2, 1.2, 1, 1])
        horizon = c[0].slider("Horizon (GWs)", 1, 8, config.HORIZON,
                              help="How many gameweeks ahead to plan for. Each week further out counts "
                                   "for less (x0.8 per week).\n\n"
                                   "**3** is the default: it scored best in replays of past seasons, since "
                                   "forecasts fade fast and later weeks add more noise than signal.\n\n"
                                   "- **1** for a Free Hit week\n"
                                   "- **4-6** when planning a Wildcard\n"
                                   "- long enough to include a coming blank/double GW or a banked transfer\n\n"
                                   "Tip: compare 3 with 5. If this week's move is the same, it's robust.")
        choices = models.trained()
        model = c[1].selectbox("Model", choices,
                               index=choices.index(config.MODEL) if config.MODEL in choices else 0,
                               help="  ".join(f"{n}: {models.DESCRIPTIONS[n]}." for n in choices)
                               + "  Train the others with `xpfpl train --model <name>`.")
        chip_label = c[2].selectbox("Chip active this GW", list(ACTIVE_CHIPS),
                                    help="Chips you've activated aren't visible via the API until the deadline.")
        ft_choice = c[3].selectbox("Free transfers", [f"Estimated ({me.free_transfers})", 0, 1, 2, 3, 4, 5])
        bank = c[4].number_input("Bank (£m)", value=float(me.bank), step=0.1, format="%.1f")
        max_hits = c[5].number_input("Max transfer penalties", 0, 5, config.MAX_HITS,
                                     help=f"Extra transfers allowed beyond your free ones, at -{config.HIT_COST} each.")
        c = st.columns([1, 1, 2])
        weekly = c[0].checkbox("Plan transfers week by week", value=config.PLAN_TRANSFERS,
                               help="Plan a squad for every gameweek in the horizon, banking free "
                                    "transfers as the game does, instead of holding one squad throughout.")
        value_prices = c[1].checkbox("Value price rises", value=bool(config.PRICE_WEIGHT),
                                     help="Nudge the optimiser towards players expected to rise in price.")
        c = st.columns(2)
        options = sorted(elements.index, key=lambda p: -elements.at[p, "total_points"])
        must_have = c[0].multiselect("Always pick", options, format_func=player_label.get,
                                     help="Force players into the squad.")
        banned = c[1].multiselect("Never pick", options, format_func=player_label.get,
                                  help="Injury news, rotation worries, or just a hunch.")
        st.form_submit_button("Update plan", type="primary")

    active = ACTIVE_CHIPS[chip_label]
    ft = me.free_transfers if isinstance(ft_choice, str) else ft_choice
    players, gameweeks = predictions(horizon, model, stamp())
    gw = gameweeks[0]
    labels = fixture_labels(bs, fx, gameweeks)
    kwargs = dict(current_squad=me.squad, bank=round(bank, 1), must_have=tuple(must_have), banned=tuple(banned))
    planning = dict(plan_transfers=bool(weekly), pool_size=config.POOL_SIZE if weekly else None,
                    price_weight=config.PRICE_WEIGHT or 1.0 if value_prices else 0.0)

    if active in ("wildcard", "freehit"):
        plan_gws = (gw,) if active == "freehit" else tuple(gameweeks)
        plan = run_solve(horizon, model, plan_gws, stamp(), **kwargs, unlimited_transfers=True)
        hold = run_solve(horizon, model, plan_gws, stamp(), **kwargs, free_transfers=0, max_hits=0)
    else:
        extra = {"triple_captain_gw": gw} if active == "3xc" else {"bench_boost_gw": gw} if active == "bboost" else {}
        plan_gws = tuple(gameweeks)
        plan = run_solve(horizon, model, plan_gws, stamp(), **kwargs, **planning, free_transfers=ft,
                         max_hits=int(max_hits), **extra)
        hold = run_solve(horizon, model, plan_gws, stamp(), **kwargs, free_transfers=0, max_hits=0, **extra)
    gain = sum(plan.xp.values()) - sum(hold.xp.values()) - config.HIT_COST * plan.total_hits

    st.subheader(f"{me.name}: plan for GW{plan_gws[0]}" + (f"-{plan_gws[-1]}" if len(plan_gws) > 1 else ""))
    m = st.columns(5)
    m[0].metric("Transfers", len(plan.transfers_in))
    m[1].metric("Transfer penalties", f"-{config.HIT_COST * plan.hits}" if plan.hits else "0")
    m[2].metric(f"xP gain vs no transfers ({len(plan_gws)} GW)", f"{gain:+.2f}",
                help="The whole plan, including the transfers pencilled in for later weeks, "
                     "against keeping this squad and never transferring again.")
    m[3].metric(f"GW{gw} xP", f"{plan.xp[gw]:.2f}")
    m[4].metric("Bank after", f"£{plan.budget_left:.1f}m")
    if active:
        st.info(f"{chip_label} active for GW{gw}. "
                + ("Unlimited free transfers, planned from your squad at the last deadline. Transfers you've "
                   "already made don't show until the deadline, so compare them with the list below."
                   if active in ("wildcard", "freehit") else "The captain/bench choices below account for it."))

    if plan.transfers_in:
        xp_cols = [f"xp_{g}" for g in plan_gws]
        outs = sorted(plan.transfers_out, key=lambda p: players.at[p, "position"])
        ins = sorted(plan.transfers_in, key=lambda p: players.at[p, "position"])
        st.dataframe(pd.DataFrame({
            "Out": [players.at[p, "name"] for p in outs],
            "Sell £": [me.squad[p] for p in outs],
            "Out xP": [players.loc[p, xp_cols].sum() for p in outs],
            "In": [f"{players.at[p, 'name']} ({players.at[p, 'team_name']})" for p in ins],
            "Buy £": [players.at[p, "price"] for p in ins],
            "In xP": [players.loc[p, xp_cols].sum() for p in ins],
        }), hide_index=True, width="stretch",
            column_config={"Sell £": st.column_config.NumberColumn(format="%.1f"),
                           "Buy £": st.column_config.NumberColumn(format="%.1f"),
                           "Out xP": st.column_config.NumberColumn(format=POINTS),
                           "In xP": st.column_config.NumberColumn(format=POINTS)})
    else:
        st.success("No transfers: roll the free transfer." if not active else "Keep the current squad.")

    if plan.future_transfers:
        st.markdown("**Planned for the following weeks** (a route, not a commitment - it is re-planned "
                    "every week from fresh predictions)")
        st.dataframe(pd.DataFrame([{
            "GW": g,
            "Out": ", ".join(players.at[p, "name"] for p in moves["out"]),
            "In": ", ".join(f"{players.at[p, 'name']} ({players.at[p, 'team_name']})" for p in moves["in"]),
            "Penalty": f"-{config.HIT_COST * moves['hits']}" if moves["hits"] else "",
        } for g, moves in sorted(plan.future_transfers.items())]), hide_index=True, width="stretch")
    for note in plan.notes:
        st.caption(note)

    view_gw = st.segmented_control("Show team for", [f"GW{g}" for g in plan_gws], default=f"GW{gw}")
    view_gw = int((view_gw or f"GW{gw}")[2:])
    st.caption(f"Formation {plan.formation(players, view_gw)} · XI xP {plan.xp[view_gw]:.2f} "
               f"(captain counted twice) · bench xP {plan.bench_xp[view_gw]:.2f}. Name in club colours, "
               "fixture shaded by FPL difficulty (darker is harder), orange/red badge = chance of playing, "
               "yellow line = the club's cup or European match that week (UK day). "
               + ("Later weeks use that week's planned squad." if plan.future_transfers
                  else "Later weeks assume the same squad."))
    show_pitch(players, plan, view_gw, labels, fixture_difficulty(fx, gameweeks),
               midweek_badges(season, tuple(gameweeks)))

    st.markdown("**Squad xP by gameweek**")
    squad_gw = plan.squads.get(view_gw, plan.squad)
    grid = players.loc[squad_gw, ["name", "team_name", "team", "position"]].copy()
    grid["position"] = grid["position"].map(config.POSITIONS)
    for g in gameweeks:
        grid[f"GW{g}"] = players.loc[squad_gw, f"xp_{g}"]
        grid[f"GW{g} fixture"] = [labels.get((players.at[p, "team"], g), "-") for p in squad_gw]
    grid["Total"] = players.loc[squad_gw, [f"xp_{g}" for g in gameweeks]].sum(axis=1)
    grid = grid.sort_values(["position", "Total"], ascending=[True, False], key=lambda s: s.map(
        {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}) if s.name == "position" else s)
    grid_team_ids = grid["team"].copy()
    grid = grid.drop(columns=["team"])
    st.dataframe(style_team_column(
                 grid.rename(columns={"name": "Player", "team_name": "Team", "position": "Pos"}),
                 grid_team_ids),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format=POINTS)
                                for c in grid.columns if c.startswith("GW") and "fixture" not in c or c == "Total"})

    st.markdown("**Chips**")
    if active:
        st.write(f"{chip_label} is active, so no other chip can be played in GW{gw}.")
    else:
        available = chips.available_chips(bs, me.chips_used, gw)
        if not available:
            st.write("No chips available this gameweek.")
        else:
            advice = run_chip_advice(horizon, model, stamp(), int(team_id), tuple(sorted(available)),
                                     **kwargs, **planning, free_transfers=ft, max_hits=int(max_hits))
            st.dataframe(pd.DataFrame([{
                "Chip": chips.CHIP_NAMES[a.chip], "Advice": "PLAY" if a.recommended else "save",
                "Gain": a.gain, "Threshold": a.threshold, "Why": a.reason,
                "Expires": f"after GW{available[a.chip]['stop_event']}",
            } for a in advice]), hide_index=True, width="stretch",
                column_config={"Gain": st.column_config.NumberColumn(format=POINTS),
                               "Threshold": st.column_config.NumberColumn(format=POINTS)})
            st.caption("Thresholds come from `xpfpl tune`, which scores candidate values by replaying "
                       "whole seasons (config.py). The Wildcard gain may be overstated because it is compared "
                       "with holding your squad for the whole horizon.")


# ---------------------------------------------------------------- gameweek review

with tab_review:
    if not last_gw:
        st.info("No gameweek has finished yet.")
    else:
        _, history, _ = manager(int(team_id))
        played = [h["event"] for h in history["current"]]
        rgw = st.selectbox("Gameweek", sorted(played, reverse=True), format_func=lambda g: f"GW{g}")
        points, r, best, best_plan = gw_review(int(team_id), rgw)
        picks, h, ev = r["picks"], r["history"], events[rgw]

        m = st.columns(6)
        m[0].metric("Points", h["points"], f"{h['points'] - ev['average_entry_score']:+d} vs average")
        m[1].metric("GW average", ev["average_entry_score"])
        m[2].metric("Highest", ev["highest_score"])
        m[3].metric("GW rank", f"{h['rank']:,}" if h.get("rank") else "-")
        m[4].metric("Transfer penalties", f"-{h['event_transfers_cost']}" if h["event_transfers_cost"] else "0")
        m[5].metric("Chip", chips.CHIP_NAMES.get(r["chip"], "-"))

        net = h["points"] + h["event_transfers_cost"]  # points before transfer penalties
        xp, xp_source = gw_xp(season, rgw, model)
        if len(xp):
            forecast = review.team_forecast(picks, xp)
            f = st.columns(3)
            f[0].metric("Forecast for your team", f"{forecast:.1f}",
                        help="The model's xP for your XI as picked: captain doubled (tripled with Triple Captain), "
                             "bench counted only with Bench Boost")
            f[1].metric("Scored", net, help="Points before transfer penalties, auto-subs included")
            f[2].metric("Scored − forecast", f"{net - forecast:+.1f}")
        st.markdown(f"**Hindsight:** the best XI and captain from the same 15 players would have scored "
                    f"**{best:.0f}**, against your {net} before transfer penalties (**{best - net:.0f}** left on the table).")
        best_xi, best_cap = set(best_plan.lineups[rgw]), best_plan.captains[rgw]
        your_xi = set(picks.index[picks["multiplier"] > 0])
        your_cap = picks.index[picks["is_captain"]][0]
        notes = []
        if best_cap != your_cap:
            notes.append(f"Should have captained {picks.at[best_cap, 'name']} ({picks.at[best_cap, 'points']:.0f}) "
                         f"rather than {picks.at[your_cap, 'name']} ({picks.at[your_cap, 'points']:.0f})")
        starts = sorted(best_xi - your_xi, key=lambda p: -picks.at[p, "points"])
        benches = sorted(your_xi - best_xi, key=lambda p: picks.at[p, "points"])
        for on, off in zip(starts, benches):
            if picks.at[on, "points"] != picks.at[off, "points"]:  # equal points: the swap changes nothing
                notes.append(f"Should have started {picks.at[on, 'name']} ({picks.at[on, 'points']:.0f}) instead of "
                             f"{picks.at[off, 'name']} ({picks.at[off, 'points']:.0f})")
        if notes:
            st.markdown("\n".join(f"- {n}" for n in notes))
        if r["auto_subs"]:
            st.caption("Auto-subs: " + ", ".join(
                f"{player_label[s['element_in']].split(' (')[0]} on for {player_label[s['element_out']].split(' (')[0]}"
                for s in r["auto_subs"]))

        midweek_mins = cups.player_minutes_before(season, rgw, elements.reset_index()[["id", "code", "team_code"]])
        table = picks.assign(
            Role=["C" if c else "VC" if v else "" for c, v in zip(picks["is_captain"], picks["is_vice_captain"])],
            Status=["Starting" if m else "Bench" for m in picks["multiplier"]],
            Pos=picks["pos"].map(config.POSITIONS),
            xP=xp.reindex(picks.index).fillna(0.0),
            Midweek=midweek_mins.reindex(picks.index),
        )
        table["Points - xP"] = table["points"] - table["xP"]
        review_table = table[["Status", "Pos", "name", "team_name", "team", "Role", "minutes", "Midweek", "xP", "points",
                      "Points - xP", "counted"]].rename(columns={"name": "Player", "team_name": "Team",
                                            "team": "Team ID", "minutes": "Mins",
                                            "points": "Points", "counted": "Counted"})
        review_team_ids = review_table.pop("Team ID")
        st.dataframe(style_team_column(review_table, review_team_ids),
                     hide_index=True, width="stretch",
                     column_config={"xP": st.column_config.NumberColumn(format=POINTS),
                                    "Points - xP": st.column_config.NumberColumn(format="%+.2f"),
                                    **{c: st.column_config.NumberColumn(format="%d")
                                       for c in ("Mins", "Midweek", "Points", "Counted")}})
        st.caption(f"{xp_source_note([xp_source] if xp_source else [])} Model: {model}. Midweek: minutes "
                   "in the club's cup or European match before this gameweek (empty if the club had none).")

        results = midweek_results(season, rgw)
        if len(results):
            club = {t["code"]: t["name"] for t in bs["teams"]}
            name = lambda code, fallback: club.get(int(code), fallback) if pd.notna(code) else fallback  # noqa: E731
            with st.expander(f"Midweek before GW{rgw}: {len(results)} cup and European matches"):
                st.markdown("\n".join(
                    f"- {cups.NAMES.get(m.tournament, m.tournament)}: {name(m.home_code, m.home_name)} "
                    f"{m.home_score:.0f}–{m.away_score:.0f} {name(m.away_code, m.away_name)}"
                    for m in results.itertuples()))

        st.markdown(f"**Top scorers in GW{rgw}**")
        top = points.join(xp.rename("xP")).nlargest(15, "points")
        top_table = pd.DataFrame({
            "Player": [player_label[p] for p in top.index],
            "Team": [team_short[elements.at[p, "team"]] for p in top.index],
            "Points": top["points"], "xP": top["xP"], "Mins": top["minutes"],
            "Owned": ["yes" if p in picks.index else "" for p in top.index],
            "Selected by %": pd.to_numeric(elements["selected_by_percent"], errors="coerce").reindex(top.index),
        })
        top_team_ids = elements.loc[top.index, "team"]
        st.dataframe(style_team_column(top_table, top_team_ids), hide_index=True, width="stretch",
                     column_config={"xP": st.column_config.NumberColumn(format=POINTS),
                                    "Points": st.column_config.NumberColumn(format="%d"),
                                    "Mins": st.column_config.NumberColumn(format="%d"),
                                    "Selected by %": st.column_config.NumberColumn(format=DECIMAL)})

        played_xp = points.join(xp.rename("xP"), how="inner")
        played_xp = played_xp[played_xp["minutes"] > 0]
        if len(played_xp):
            err = played_xp["xP"] - played_xp["points"]
            st.caption(f"Model accuracy in GW{rgw} over {len(played_xp)} players who played: "
                       f"MAE {err.abs().mean():.2f}, RMSE {(err ** 2).mean() ** 0.5:.2f} points.")


# ---------------------------------------------------------------- my season

with tab_season:
    info, history, transfers = manager(int(team_id))
    hist = pd.DataFrame(history["current"])
    if hist.empty:
        st.info("No gameweeks played yet.")
    else:
        hist["average"] = hist["event"].map(lambda g: events[g]["average_entry_score"])
        latest = hist.iloc[-1]
        m = st.columns(5)
        m[0].metric("Total points", int(latest["total_points"]))
        m[1].metric("Overall rank", f"{int(latest['overall_rank']):,}")
        m[2].metric("Team value", f"£{latest['value'] / 10:.1f}m")
        m[3].metric("Bank", f"£{latest['bank'] / 10:.1f}m")
        m[4].metric("Points on bench", int(hist["points_on_bench"].sum()))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Points per gameweek**")
            base = alt.Chart(hist).encode(x=alt.X("event:O", title="Gameweek"))
            bars = base.mark_bar(color=SERIES, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=24).encode(
                y=alt.Y("points:Q", title="Points"),
                tooltip=[alt.Tooltip("event:O", title="GW"), alt.Tooltip("points:Q", title="Points"),
                         alt.Tooltip("average:Q", title="GW average"),
                         alt.Tooltip("event_transfers_cost:Q", title="Transfer penalty")])
            avg = base.mark_line(color=MUTED, strokeDash=[4, 4], strokeWidth=2).encode(y="average:Q")
            dots = base.mark_point(color=MUTED, filled=True, size=40).encode(
                y="average:Q", tooltip=[alt.Tooltip("event:O", title="GW"),
                                        alt.Tooltip("average:Q", title="GW average")])
            st.altair_chart(bars + avg + dots, width="stretch")
        with c2:
            st.markdown("**Overall rank**")
            rank = alt.Chart(hist).mark_line(color=SERIES, strokeWidth=2, point=alt.OverlayMarkDef(
                color=SERIES, size=64, filled=True)).encode(
                x=alt.X("event:O", title="Gameweek"),
                y=alt.Y("overall_rank:Q", title="Overall rank", scale=alt.Scale(reverse=True, zero=False),
                        axis=alt.Axis(format="~s")),
                tooltip=[alt.Tooltip("event:O", title="GW"), alt.Tooltip("overall_rank:Q", title="Rank", format=",")])
            st.altair_chart(rank, width="stretch")

        forecasts = team_forecasts(int(team_id), tuple(int(g) for g in hist["event"]), season, model, stamp())
        hist = hist.merge(forecasts, on="event", how="left")
        hist["scored"] = hist["points"] + hist["event_transfers_cost"]      # before transfer penalties
        hist["vs_forecast"] = hist["scored"] - hist["forecast"]
        compared = hist[hist["forecast"].notna()]
        if len(compared):
            st.markdown("**Forecast against points scored**")
            f = st.columns(3)
            f[0].metric("Forecast, all season", f"{compared['forecast'].sum():.0f}",
                        help=f"{len(compared)} gameweeks: the model's xP for your team as picked each week")
            f[1].metric("Scored", f"{compared['scored'].sum():.0f}", help="Before transfer penalties")
            f[2].metric("Scored − forecast", f"{compared['vs_forecast'].sum():+.0f}",
                        f"{compared['vs_forecast'].mean():+.1f} a gameweek", delta_color="off")
            long = pd.concat([compared.assign(series="Scored", value=compared["scored"]),
                              compared.assign(series="Forecast", value=compared["forecast"])])
            chart = alt.Chart(long).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("event:O", title="Gameweek"), xOffset=alt.XOffset("series:N", sort=["Forecast", "Scored"]),
                y=alt.Y("value:Q", title="Points"),
                color=alt.Color("series:N", sort=["Forecast", "Scored"], title=None, legend=alt.Legend(orient="top"),
                                scale=alt.Scale(domain=["Forecast", "Scored"], range=[FORECAST, SERIES])),
                tooltip=[alt.Tooltip("event:O", title="GW"), alt.Tooltip("forecast:Q", title="Forecast", format=".1f"),
                         alt.Tooltip("scored:Q", title="Scored"), alt.Tooltip("vs_forecast:Q", title="Scored − forecast",
                                                                              format="+.1f")])
            st.altair_chart(chart, width="stretch")
            st.caption("Forecast: the model's xP for your XI as you picked it, captain doubled (tripled with Triple "
                       "Captain), bench only with Bench Boost. Scored: your points before transfer penalties. "
                       + xp_source_note(compared["source"]))

        st.dataframe(hist[["event", "points", "forecast", "vs_forecast", "average", "rank", "overall_rank",
                           "event_transfers", "event_transfers_cost", "points_on_bench", "bank", "value"]].assign(
            bank=hist["bank"] / 10, value=hist["value"] / 10).rename(columns={
                "event": "GW", "points": "Points", "forecast": "Forecast", "vs_forecast": "vs forecast",
                "average": "GW avg", "rank": "GW rank",
                "overall_rank": "Overall rank", "event_transfers": "Transfers", "event_transfers_cost": "Penalty",
                "points_on_bench": "Bench pts", "bank": "Bank £m", "value": "Value £m"}),
            hide_index=True, width="stretch",
            column_config={"Forecast": st.column_config.NumberColumn(format="%.1f"),
                           "vs forecast": st.column_config.NumberColumn(format="%+.1f",
                                                                        help="Points before penalties minus the forecast")})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Chips**")
            used = {(c["name"], c["event"]) for c in history["chips"]}
            rows = []
            for w in bs["chips"]:
                when = next((e for n, e in used if n == w["name"] and w["start_event"] <= e <= w["stop_event"]), None)
                rows.append({"Chip": chips.CHIP_NAMES.get(w["name"], w["name"]),
                             "Window": f"GW{w['start_event']}-{w['stop_event']}",
                             "Status": f"Used GW{when}" if when else "Available"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("A chip activated for the upcoming GW shows here only after the deadline.")
        with c2:
            st.markdown("**Transfers**")
            if transfers:
                st.dataframe(pd.DataFrame([{
                    "GW": t["event"],
                    "Out": player_label.get(t["element_out"], t["element_out"]).split(" (")[0],
                    "Sold £": t["element_out_cost"] / 10,
                    "In": player_label.get(t["element_in"], t["element_in"]).split(" (")[0],
                    "Bought £": t["element_in_cost"] / 10,
                } for t in transfers]), hide_index=True, width="stretch")
            else:
                st.write("No transfers yet.")


# ---------------------------------------------------------------- players and fixtures

with tab_players:
    players, gameweeks = predictions(horizon, model, stamp())
    c = st.columns([1.2, 1.5, 1, 1.3, 0.8])
    pos_f = c[0].multiselect("Position", list(config.POSITIONS.values()))
    team_f = c[1].multiselect("Team", sorted(team_short.values()))
    max_price = c[2].slider("Max price", 3.5, 16.0, 16.0, 0.5)
    search = c[3].text_input("Search name")
    avail_only = c[4].checkbox("Fit only", value=True, help="Hide flagged or departed players")

    table = players.copy()
    table["Pos"] = table["position"].map(config.POSITIONS)
    if pos_f:
        table = table[table["Pos"].isin(pos_f)]
    if team_f:
        table = table[table["team_name"].isin(team_f)]
    table = table[table["price"] <= max_price]
    if search:
        table = table[table["name"].str.contains(search, case=False, na=False)]
    if avail_only:
        table = table[table["status"] == "a"]
    xp_cols = [f"xp_{g}" for g in gameweeks]
    minutes_cols = [c for c in ("xmins", "p_play") if c in table]    # only minutes-aware models
    if "rotation" in table and table["rotation"].notna().any():       # next GW's midweek factor
        table = table.assign(midweek=[f"×{f:.2f} ({cups.ROTATION_LABELS.get(g, g)})" if isinstance(g, str) else ""
                                      for f, g in zip(table["rotation_factor"], table["rotation"])])
        minutes_cols += ["midweek"]
    if "market_out" in table:                                         # next GW's betting odds, when listed
        table = table.assign(market_says=np.where(table["market_out"].fillna(False).astype(bool), "Likely out", ""),
                             mkt_anytime=table["mkt_anytime"] * 100)
        minutes_cols += ["mkt_anytime", "market_says"]
    shown = table.sort_values("xp_total", ascending=False)[
        ["name", "team_name", "team", "Pos", "price", "selected_by", "chance"] + minutes_cols + xp_cols
        + ["xp_total"]]
    shown["xp_per_m"] = shown["xp_total"] / shown["price"]
    shown_team_ids = shown["team"].copy()
    shown = shown.drop(columns=["team"])
    shown = shown.rename(columns={
        "name": "Player", "team_name": "Team", "team": "Team ID", "price": "£m", "selected_by": "Sel %",
        "chance": "Availability %", "xmins": f"xMins GW{gameweeks[0]}", "p_play": "P(plays)",
        "mkt_anytime": "Scorer odds %", "market_says": "Market says", "midweek": f"Midweek (GW{gameweeks[0]})",
        "xp_total": "Total xP", "xp_per_m": "xP per £m", **{f"xp_{g}": f"GW{g}" for g in gameweeks}})
    st.dataframe(style_team_column(shown, shown_team_ids),
        hide_index=True, width="stretch", height=420,
        column_config={**{f"GW{g}": st.column_config.NumberColumn(format=POINTS) for g in gameweeks},
                       "Total xP": st.column_config.NumberColumn(format=POINTS),
                       "xP per £m": st.column_config.NumberColumn(format="%.2f"),
                       f"xMins GW{gameweeks[0]}": st.column_config.NumberColumn(format="%.0f"),
                       "P(plays)": st.column_config.NumberColumn(format="%.2f"),
                       "Sel %": st.column_config.NumberColumn(format=DECIMAL),
                       "Availability %": st.column_config.NumberColumn(format=DECIMAL),
                       "Scorer odds %": st.column_config.NumberColumn(format=DECIMAL),
                       "£m": st.column_config.NumberColumn(format="%.1f")})
    st.caption(f"{len(shown)} players · xP is per gameweek, scaled by FPL's injury flag "
               "(assumes +25% recovery per week) and summed over double gameweeks. 'Market says: Likely out' means "
               "the betting market prices him at 6% or less to score next gameweek, which almost always means "
               "he's been ruled out; that week's xP is cut to 10%. 'Midweek': his club played a cup or European "
               "match before the next gameweek, and his xP was multiplied by what his own minutes in it have meant "
               "(see the Guide's glossary).")

    st.markdown("**A player's season: forecast against points**")
    season_rows = load_matches()
    season_rows = season_rows[season_rows["season"] == season]
    if season_rows.empty:
        st.caption("No gameweek of this season has been played yet.")
    else:
        default = int(players["xp_total"].idxmax()) if len(players) else None
        options = sorted(player_label, key=lambda p: player_label[p])
        chosen = st.selectbox("Player", options, index=options.index(default) if default in options else 0,
                              format_func=lambda p: player_label[p], key="player_season")
        actual = (season_rows[season_rows["element"] == chosen].groupby("gw")[["total_points", "minutes"]].sum())
        xp_all = season_xp(season, model, stamp())
        mine = xp_all[xp_all["element"] == chosen].set_index("gw")["xp"]
        gws_played = sorted(int(g) for g in season_rows["gw"].unique())
        detail = pd.DataFrame({"gw": gws_played}).assign(
            points=lambda d: d["gw"].map(actual["total_points"]).fillna(0),
            minutes=lambda d: d["gw"].map(actual["minutes"]).fillna(0),
            xp=lambda d: d["gw"].map(mine))
        detail["diff"] = detail["points"] - detail["xp"]
        known = detail[detail["xp"].notna()]
        f = st.columns(4)
        f[0].metric("Points", f"{detail['points'].sum():.0f}", help=f"{len(detail)} gameweeks")
        f[1].metric("Forecast (xP)", f"{known['xp'].sum():.1f}" if len(known) else "-")
        f[2].metric("Points − xP", f"{known['diff'].sum():+.1f}" if len(known) else "-")
        f[3].metric(f"Forecast for GW{gameweeks[0]}", f"{players.at[chosen, f'xp_{gameweeks[0]}']:.2f}"
                    if chosen in players.index else "-")
        base = alt.Chart(detail).encode(x=alt.X("gw:O", title="Gameweek"))
        bars = base.mark_bar(color=SERIES, cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=22).encode(
            y=alt.Y("points:Q", title="Points"),
            tooltip=[alt.Tooltip("gw:O", title="GW"), alt.Tooltip("points:Q", title="Points"),
                     alt.Tooltip("xp:Q", title="xP", format=".2f"), alt.Tooltip("minutes:Q", title="Minutes")])
        line = base.mark_line(color=FORECAST, strokeWidth=2, point=alt.OverlayMarkDef(color=FORECAST, size=50,
                                                                                      filled=True)).encode(
            y="xp:Q", tooltip=[alt.Tooltip("gw:O", title="GW"), alt.Tooltip("xp:Q", title="xP", format=".2f")])
        st.altair_chart(bars + line, width="stretch")
        st.caption("Bars: points scored. Orange: the model's xP for that gameweek. "
                   + xp_source_note(xp_all.loc[xp_all["gw"].isin(known["gw"]), "source"].unique()))

    st.markdown("**Fixtures ahead**")
    rows = {}
    for f in fx:
        if f["event"] in gameweeks:
            for team, opp, home, fdr in ((f["team_h"], f["team_a"], "H", f["team_h_difficulty"]),
                                         (f["team_a"], f["team_h"], "A", f["team_a_difficulty"])):
                rows.setdefault(team, {}).setdefault(f["event"], []).append((f"{team_short[opp]} ({home})", fdr))
    badge = midweek_badges(season, tuple(gameweeks))
    ticker = pd.DataFrame({f"GW{g}": {team_short[t]: (f"[{badge[(t, g)]}] " if (t, g) in badge else "")
                                       + (", ".join(l for l, _ in v.get(g, [])) or "blank")
                                       for t, v in rows.items()} for g in gameweeks})
    difficulty = pd.DataFrame({f"GW{g}": {team_short[t]: (sum(d for _, d in v[g]) / len(v[g]) if g in v else 0)
                                           for t, v in rows.items()} for g in gameweeks})
    ticker["Average Fixture Difficulty Rating"] = difficulty.replace(0, pd.NA).mean(axis=1).round(2)
    ticker = ticker.sort_values("Average Fixture Difficulty Rating")
    difficulty = difficulty.reindex(ticker.index)

    def colour(col: pd.Series) -> list[str]:
        if col.name == "Average Fixture Difficulty Rating":
            return [""] * len(col)
        return [f"background-color: {FDR_BG.get(round(d), '#f0efec')}; color: {FDR_FG.get(round(d), '#0b0b0b')}"
                for d in difficulty[col.name]]

    st.caption("FPL fixture difficulty. [UCL Tue] and the like: the club's cup or European match in the week "
               "before that gameweek (UCL Champions League, UEL Europa League, UECL Conference League, EFL EFL Cup; "
               "days in UK time).")
    legend = st.columns(5)
    for difficulty_level, slot in enumerate(legend, start=1):
        slot.markdown(
            f"<div style='background:{FDR_BG[difficulty_level]};color:{FDR_FG[difficulty_level]};"
            f"padding:0.25rem 0.4rem;text-align:center;border-radius:2px;'>"
            f"{difficulty_level} · {'Easiest' if difficulty_level == 1 else 'Hardest' if difficulty_level == 5 else ''}</div>",
            unsafe_allow_html=True)
    st.dataframe(ticker.style.apply(colour).format({"Average Fixture Difficulty Rating": "{:.2f}"}), width="stretch",
                 height=38 * (len(ticker) + 1))


# ---------------------------------------------------------------- markets

with tab_markets:
    market_bs, market_fx = live_data()
    market_view.render(market_bs, market_fx)
