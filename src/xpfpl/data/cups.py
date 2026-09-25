"""Midweek cup and European matches, for rotation risk.

A club with a Champions League match on Tuesday may rest its stars on Saturday, or take them off
at 60 minutes. vaastav and the FPL API only know Premier League matches, so this reads the other
competitions from FPL-Core-Insights (github.com/olbauday/FPL-Core-Insights), which files every
match an EPL club plays (Champions/Europa/Conference League, EFL Cup, FA Cup...) under the FPL
gameweek it falls before, with per-player minutes. Its club ids are FPL `team_code`s and its
players carry the FPL `player_code`, so both link straight to our `team_code` / `code`.

Coverage: 2025-26 onwards (2024-25 there holds Premier League matches only), fixtures listed
weeks ahead. Older seasons get `cup_era = 0` and zeros, the same trick as `dc_era`.

archive/cups/<season>/fixtures/gwNN.parquet   one row per EPL club per non-EPL match
archive/cups/<season>/minutes/gwNN.parquet    one row per player per played non-EPL match

`rest_features` turns these into features for each Premier League player-match: how recently
the club played another competition (and how big), whether another one follows soon after, and
how many minutes the player himself played in it. `fit` / `adjust` turn the player's own midweek
role into a factor on next week's xP (see "adjusting xP" below); `predict.py` applies it.
"""

import io
import re
import json

import numpy as np
import pandas as pd
import requests

from xpfpl import config
from xpfpl.data import archive

SOURCE = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data"
FIRST_SEASON = "2025-26"
RAW_DIR = config.RAW_DIR / "coreinsights"
ARCHIVE = config.ARCHIVE_DIR / "cups"
EPL = "prem"
# How much a competition matters to a club's selection, roughly: a Champions League night is
# what gets a Saturday line-up rotated. Unknown competitions count as 1.
LEVELS = {"champions-league": 3, "europa-league": 2, "conference-league": 1, "efl-cup": 1,
          "fa-cup": 1, "uefa-super-cup": 2, "community-shield": 0, "friendlies": 0}
WINDOW_DAYS = 6.0     # a cup match counts as "before" / "after" an EPL match within this many days
FEATURES = ["cup_era", "cup_before", "cup_before_days", "cup_before_level",
            "cup_after", "cup_after_days", "cup_after_level", "cup_mins_before", "cup_started_before"]

_session = requests.Session()
_session.headers["User-Agent"] = "xpfpl (personal research project)"


def _source_season(season: str) -> str:
    """'2025-26' -> '2025-2026', the source's folder name."""
    start = int(season[:4])
    return f"{start}-{start + 1}"


def _csv(season: str, gw: int, name: str, refresh: bool) -> pd.DataFrame | None:
    """One of the source's per-gameweek CSVs, cached in data/raw (None if it isn't there)."""
    path = RAW_DIR / season / f"gw{gw:02d}" / f"{name}.csv"
    if refresh or not path.exists():
        url = f"{SOURCE}/{_source_season(season)}/By%20Gameweek/GW{gw}/{name}.csv"
        resp = _session.get(url, timeout=60)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    return pd.read_csv(io.BytesIO(path.read_bytes()), encoding="utf-8", encoding_errors="replace")


def club_fixtures(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """The source's fixtures.csv -> one row per EPL club per non-EPL match (the other side of a
    European tie has no FPL code, so it is dropped)."""
    cups = raw[raw["tournament"] != EPL]
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        s = cups[cups[f"{side}_team"].notna()]
        sides.append(pd.DataFrame({
            "season": season,
            "gw": s["gameweek"].astype(int),
            "match_id": s["match_id"],
            "tournament": s["tournament"],
            "kickoff_time": pd.to_datetime(s["kickoff_time"], utc=True),
            "team_code": s[f"{side}_team"].astype(int),
            "opp_code": s[f"{other}_team"].astype("Int64"),
            "was_home": side == "home",
            "goals_for": s[f"{side}_score"],
            "goals_against": s[f"{other}_score"],
            "finished": s["finished"].astype(bool),
        }))
    return pd.concat(sides, ignore_index=True).sort_values(["kickoff_time", "team_code"], ignore_index=True)


def player_minutes(stats: pd.DataFrame, players: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """playermatchstats.csv -> minutes per player (by FPL `code`) in this GW's non-EPL matches."""
    cup_ids = set(fixtures["match_id"])
    s = stats[stats["match_id"].isin(cup_ids)]
    s = s.merge(players[["player_id", "player_code", "team_code"]], on="player_id", how="inner")
    return pd.DataFrame({
        "match_id": s["match_id"],
        "code": s["player_code"].astype(int),
        "team_code": s["team_code"].astype(int),
        "minutes": s["minutes_played"].fillna(0).astype(int),
        "started": s["start_min"].fillna(-1).eq(0),
    }).reset_index(drop=True)


def fetch(season: str, gameweeks=range(1, 39), refresh: bool = False) -> int:
    """Download a season's non-EPL fixtures and minutes into archive/cups. Returns files written.

    Played gameweeks are cached in data/raw; `refresh` re-downloads them (the current and future
    ones are always re-read, since kick-off times move and scores arrive)."""
    written = 0
    now = pd.Timestamp.now(tz="UTC")
    for gw in gameweeks:
        raw = _csv(season, gw, "fixtures", refresh)
        if raw is None or raw.empty:
            continue
        if not raw["finished"].astype(bool).all() and not refresh:
            raw = _csv(season, gw, "fixtures", refresh=True)
        fx = club_fixtures(raw, season)
        written += archive.write(fx, ARCHIVE / season / "fixtures" / f"gw{gw:02d}.parquet")
        played = fx[fx["finished"] & (fx["kickoff_time"] < now)]
        if played.empty:
            continue
        stats = _csv(season, gw, "playermatchstats", refresh)
        players = _csv(season, gw, "players", refresh)
        if stats is None or players is None:
            continue
        mins = player_minutes(stats, players, played)
        if len(mins):
            written += archive.write(mins, ARCHIVE / season / "minutes" / f"gw{gw:02d}.parquet")
    return written


def _stack(kind: str) -> pd.DataFrame | None:
    parts = sorted(ARCHIVE.glob(f"*/{kind}/gw*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else None


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(club fixtures, player minutes) for every archived season; empty frames if none."""
    fx = _stack("fixtures")
    mins = _stack("minutes")
    if fx is None:
        fx = pd.DataFrame(columns=["season", "gw", "match_id", "tournament", "kickoff_time", "team_code"])
    if mins is None:
        mins = pd.DataFrame(columns=["match_id", "code", "team_code", "minutes", "started"])
    fx = fx.drop_duplicates(["match_id", "team_code"], keep="last")   # a tie refiled under a later GW
    return fx, mins


def _nearest(epl: pd.DataFrame, cups: pd.DataFrame, direction: str) -> pd.DataFrame:
    """For each EPL row, the club's nearest non-EPL match `backward` (before) or `forward`."""
    left = epl[["team_code", "kickoff_time"]].reset_index().sort_values("kickoff_time")
    right = (cups[["team_code", "kickoff_time", "match_id", "tournament"]].dropna(subset=["kickoff_time"])
             .rename(columns={"kickoff_time": "cup_time"}).sort_values("cup_time"))   # TBD ties have no time
    right["team_code"] = right["team_code"].astype(int)
    left["team_code"] = left["team_code"].astype(int)
    left["kickoff_time"] = pd.to_datetime(left["kickoff_time"], utc=True).astype(right["cup_time"].dtype)
    got = pd.merge_asof(left, right, left_on="kickoff_time", right_on="cup_time", by="team_code",
                        direction=direction, allow_exact_matches=False,
                        tolerance=pd.Timedelta(days=WINDOW_DAYS))
    return got.set_index("index").reindex(epl.index)


def rest_features(epl: pd.DataFrame, fixtures: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    """Rotation features for each EPL player-match row of `epl` (needs `season`, `team_code`,
    `code`, `kickoff_time`), indexed like it:

      cup_era             1 where the season has cup data at all (else every feature is 0)
      cup_before          1 if the club played another competition in the 6 days before
      cup_before_days     days since it (6 if none), and `cup_before_level` how big (LEVELS)
      cup_after / _days / _level   the same for the next non-EPL match within 6 days after
      cup_mins_before     the player's minutes in that match / 90 (0 if unknown or not played)
      cup_started_before  1 if he started it

    `cup_mins_before` and `cup_started_before` are only known once that match is played (NaN
    before): fine for training and for the next gameweek (the midweek match is before the
    deadline), not for weeks further ahead.
    """
    out = pd.DataFrame(index=epl.index)
    out["cup_era"] = epl["season"].isin(fixtures["season"].unique()).astype(float)
    for name, direction in (("before", "backward"), ("after", "forward")):
        near = _nearest(epl, fixtures, direction)
        days = (near["kickoff_time"] - near["cup_time"]).abs().dt.total_seconds() / 86400
        out[f"cup_{name}"] = near["match_id"].notna().astype(float)
        out[f"cup_{name}_days"] = days.fillna(WINDOW_DAYS).clip(upper=WINDOW_DAYS)
        out[f"cup_{name}_level"] = near["tournament"].map(LEVELS).fillna(
            near["match_id"].notna().astype(float))
        if name == "before":
            key = pd.DataFrame({"match_id": near["match_id"], "code": epl["code"]}, index=epl.index)
            played = key.reset_index().merge(minutes[["match_id", "code", "minutes", "started"]],
                                             on=["match_id", "code"], how="left").set_index("index")
            # A player missing from a played match's stats wasn't used (0); a match not played
            # yet leaves his minutes unknown (NaN), which `group` then ignores.
            unplayed = near["match_id"].notna() & ~near["match_id"].isin(set(minutes["match_id"]))
            out["cup_mins_before"] = (played["minutes"].fillna(0) / 90.0).clip(upper=4 / 3).mask(unplayed)
            out["cup_started_before"] = played["started"].fillna(False).astype(float).mask(unplayed)
    return out[FEATURES].astype("float32")


# ---------------------------------------------------------------- adjusting xP

# The model can't see midweek matches, so what it gets wrong around them is the adjustment: points
# scored / xP per rotation group, relative to weeks with no midweek match, on every season with cup
# data. Only the next gameweek is adjusted, from each player's own minutes in the midweek match
# (played before the deadline). For later weeks only the club's fixture list is known, and at club
# level there was no rotation penalty to apply: on 2025-26 + 2026-27 GW1-5, regulars at European
# clubs beat the model's xP *more* in weeks with a midweek match (0.99-1.09 vs 0.92), which is club
# quality, not rest. So later weeks only show the midweek matches (`predict`'s `cup_<gw>` columns).
FACTORS_PATH = config.MODELS_DIR / "cups.json"
REGULAR_MINUTES = 60      # minutes_r5 at or above this: a regular starter
FULL_MINUTES = 76         # midweek minutes at or above this: played (nearly) the whole match
PRIOR_XP = 200.0          # each factor is shrunk towards 1 as if it had this much xP at ratio 1
GROUPS = ["regular_rested", "regular_part", "regular_full", "squad_unused", "squad_played"]


def group(df: pd.DataFrame) -> pd.Series:
    """Each row's midweek rotation group (NaN where there's no played midweek match to go on).

    `df` needs `minutes_r5` and the `rest_features` columns. Regulars (5-match minutes >= 60) are
    split by their midweek minutes (0 / some / 76+); squad players by whether they got any."""
    mins = df["cup_mins_before"] * 90
    known = (df["cup_before"] > 0) & mins.notna()
    regular = df["minutes_r5"] >= REGULAR_MINUTES
    labels = pd.Series(np.nan, index=df.index, dtype=object)
    labels[known & regular & (mins == 0)] = "regular_rested"
    labels[known & regular & (mins > 0) & (mins < FULL_MINUTES)] = "regular_part"
    labels[known & regular & (mins >= FULL_MINUTES)] = "regular_full"
    labels[known & ~regular & (mins == 0)] = "squad_unused"
    labels[known & ~regular & (mins > 0)] = "squad_played"
    return labels


def fit(frame: pd.DataFrame, xp, model: str, save: bool = True) -> dict:
    """The xP factor per rotation group for `model`, from its predictions `xp` on the training
    `frame` (the rows of seasons with cup data are used). Saved under `model` in models/cups.json."""
    fixtures, minutes = load()
    rows = frame["season"].isin(fixtures["season"].unique())
    f = frame.loc[rows, ["season", "team_code", "code", "kickoff_time", "minutes_r5", "total_points"]]
    f = f.assign(xp=pd.Series(np.asarray(xp, dtype=float), index=frame.index)[rows].clip(lower=0))
    f = f.join(rest_features(f, fixtures, minutes))
    labels = group(f)
    table, bases = {}, {}
    for name in GROUPS:
        # Relative to every player of the same kind (regular / squad) whose club played midweek,
        # not to quiet weeks: clubs in Europe beat the model anyway (they're good), and that is
        # not rotation. So the factors only move xP between a club's players, by midweek role.
        kind = name.split("_")[0]
        if kind not in bases:
            same = f[labels.str.startswith(kind, na=False)]
            bases[kind] = same["total_points"].sum() / same["xp"].sum() if same["xp"].sum() > 0 else 1.0
        base = bases[kind] or 1.0
        g = f[labels == name]
        points, expected = float(g["total_points"].sum()), float(g["xp"].sum())
        factor = (points / base + PRIOR_XP) / (expected + PRIOR_XP) if expected + PRIOR_XP > 0 else 1.0
        table[name] = {"rows": len(g), "points": points, "xp": round(expected, 1),
                       "ratio": round(points / base / expected, 3) if expected else None,
                       "factor": round(factor, 3)}
    fitted = {"seasons": sorted(f["season"].unique()),
              "base_ratio": {k: round(float(v), 3) for k, v in bases.items()}, "groups": table}
    if save:
        saved = json.loads(FACTORS_PATH.read_text(encoding="utf-8")) if FACTORS_PATH.exists() else {}
        saved[model] = fitted
        FACTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        FACTORS_PATH.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    return fitted


def factors(model: str) -> dict[str, float] | None:
    """{group: factor} fitted for `model`, or None if `xpfpl train` hasn't fitted any yet."""
    if not FACTORS_PATH.exists():
        return None
    fitted = json.loads(FACTORS_PATH.read_text(encoding="utf-8")).get(model)
    return {k: v["factor"] for k, v in fitted["groups"].items()} if fitted else None


def adjust(frame: pd.DataFrame, next_gw: int, model: str) -> pd.DataFrame:
    """For upcoming fixtures (`build_future_frame`, with `kickoff_time`): each row's
    `rest_features`, its `rotation` group and the `rotation_factor` to multiply its xP by
    (1 except in the next gameweek, and 1 everywhere if no factors are fitted)."""
    fixtures, minutes = load()
    out = rest_features(frame, fixtures, minutes)
    labels = group(frame[["minutes_r5"]].join(out)).where(frame["gw"] == next_gw)
    table = factors(model) or {}
    out["rotation"] = labels
    out["rotation_factor"] = labels.map(table).fillna(1.0).astype(float)
    return out


# ---------------------------------------------------------------- for the website

INITIALS = {"ac", "aek", "afc", "aik", "as", "az", "bk", "bsc", "cd", "cf", "cfr", "cp", "csm", "fc", "ff", "fk",
            "hjk", "if", "ik", "lask", "nk", "paok", "psv", "rb", "sc", "sk", "ss", "ssc", "sv", "tsg", "ud", "vfb", "vfl"}


def _club_name(slug: str) -> str:
    """'club-brugge' -> 'Club Brugge', 'sabah-fk' -> 'Sabah FK', 'ac-milan' -> 'AC Milan'."""
    words = slug.replace("-", " ").split()
    return " ".join(w.upper() if w in INITIALS else w.title() for w in words)


def names(match_id: str, tournament: str) -> tuple[str, str]:
    """(home, away) club names from the source's match id, e.g.
    '26-27-champions-league-club-brugge-vs-aston-villa-2026-09-08' -> ('Club Brugge', 'Aston Villa')."""
    body = re.sub(r"^\d\d-\d\d-", "", match_id)
    body = body.removeprefix(f"{tournament}-")
    body = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", body)
    home, _, away = body.partition("-vs-")
    return _club_name(home), _club_name(away)


def matches(season: str) -> pd.DataFrame:
    """`season`'s cup and European matches, one row each (not one per EPL side): gw, kick-off,
    competition, both clubs (FPL `team_code` where it's an EPL club, and a name), and the score."""
    fixtures, _ = load()
    fx = fixtures[fixtures["season"] == season].drop_duplicates("match_id")
    if fx.empty:
        return pd.DataFrame(columns=["match_id", "gw", "kickoff", "tournament", "home_code", "away_code",
                                    "home_name", "away_name", "home_score", "away_score", "finished"])
    home = fx["was_home"].astype(bool)
    named = [names(m, t) for m, t in zip(fx["match_id"], fx["tournament"])]
    out = pd.DataFrame({
        "match_id": fx["match_id"],
        "gw": fx["gw"].astype(int),
        "kickoff": fx["kickoff_time"],
        "tournament": fx["tournament"],
        "home_code": fx["team_code"].where(home, fx["opp_code"]).astype("Int64"),
        "away_code": fx["opp_code"].where(home, fx["team_code"]).astype("Int64"),
        "home_name": [h for h, _ in named],
        "away_name": [a for _, a in named],
        "home_score": fx["goals_for"].where(home, fx["goals_against"]),
        "away_score": fx["goals_against"].where(home, fx["goals_for"]),
        "finished": fx["finished"].astype(bool),
    })
    # The source sometimes leaves an EPL club's code out (Fulham v AFC Bournemouth in the EFL Cup):
    # fill it from the club's name wherever that name has a code elsewhere.
    code_of = {}
    for side in ("home", "away"):
        known = out[out[f"{side}_code"].notna()]
        code_of.update(zip(known[f"{side}_name"], known[f"{side}_code"]))
    for side in ("home", "away"):
        out[f"{side}_code"] = out[f"{side}_code"].fillna(out[f"{side}_name"].map(code_of)).astype("Int64")
    return out.sort_values(["kickoff", "match_id"], na_position="last", ignore_index=True)


def gameweek_minutes(season: str, gw: int) -> pd.Series:
    """Minutes per player `code` in the cup and European matches filed under `gw` (the midweek
    before it). Players in those squads who didn't come on have 0; everyone else is missing."""
    fixtures, minutes = load()
    ids = fixtures.loc[(fixtures["season"] == season) & (fixtures["gw"] == gw), "match_id"]
    return minutes[minutes["match_id"].isin(set(ids))].groupby("code")["minutes"].sum()


def player_minutes_before(season: str, gw: int, people: pd.DataFrame) -> pd.Series:
    """Minutes per player (`people`: id, code, team_code) in the cup and European matches before
    `gw`, by id. A player at a club that played but missing from the match's squad list didn't play
    (0); a player whose club had no midweek match is left out."""
    by_code = gameweek_minutes(season, gw)
    played = matches(season)
    played = played[(played["gw"] == gw) & played["finished"]]
    clubs = set(played["home_code"].dropna()) | set(played["away_code"].dropna())
    people = people.set_index("id")
    minutes = people["code"].map(by_code)
    return minutes.where(minutes.notna() | ~people["team_code"].isin(clubs), 0.0).dropna()


# Short names for badges, and next week's rotation groups in a few words (the website has its own
# copy of both in web/src/midweek.ts).
SHORT = {"champions-league": "UCL", "europa-league": "UEL", "conference-league": "UECL", "efl-cup": "EFL",
         "fa-cup": "FA", "uefa-super-cup": "USC", "community-shield": "CS"}
NAMES = {"champions-league": "Champions League", "europa-league": "Europa League",
         "conference-league": "Conference League", "efl-cup": "EFL Cup", "fa-cup": "FA Cup",
         "uefa-super-cup": "UEFA Super Cup", "community-shield": "Community Shield"}
ROTATION_LABELS = {"regular_rested": "regular, rested", "regular_part": "regular, part of it",
                   "regular_full": "regular, whole match", "squad_unused": "squad, not used",
                   "squad_played": "squad, played"}


def badges(season: str, gameweeks) -> dict[tuple[int, int], str]:
    """(team_code, gw) -> "UCL Tue" for each club's cup or European match(es) before those
    gameweeks. The weekday is UK time: a European Tuesday night is Wednesday morning in Australia."""
    m = matches(season)
    m = m[m["gw"].isin(list(gameweeks)) & m["kickoff"].notna()]
    out: dict[tuple[int, int], list[str]] = {}
    for r in m.itertuples():
        label = f"{SHORT.get(r.tournament, r.tournament)} {pd.Timestamp(r.kickoff).tz_convert('Europe/London'):%a}"
        for code in (r.home_code, r.away_code):
            if pd.notna(code):
                out.setdefault((int(code), int(r.gw)), []).append(label)
    return {k: ", ".join(v) for k, v in out.items()}
