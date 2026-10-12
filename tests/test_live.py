from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from xpfpl import live


def test_decompose_parts_add_up_to_the_error():
    t = pd.DataFrame({
        "xp": [4.0, 4.0, 2.0, 0.2], "p_play": [0.8, 0.8, 0.5, 0.05], "minutes": [90, 0, 90, 30],
        "total_points": [12, 0, 2, 5], "position": [4, 3, 2, 3],
        "goals_scored": [2, 0, 0, 1], "assists": [0, 0, 0, 0],
        "expected_goals": [0.6, 0.0, 0.1, 0.1], "expected_assists": [0.1, 0.0, 0.0, 0.0],
    })
    parts = live.decompose(t)
    assert (parts["minutes"] + parts["finishing"] + parts["other"] - parts["error"]).abs().max() < 1e-9
    # Didn't play: the whole miss is minutes, none of it finishing.
    assert parts.loc[1, "minutes"] == pytest.approx(-4.0)
    assert parts.loc[1, "finishing"] == 0
    # Played when given 80%: +0.2 x 4 / 0.8; two goals from 0.6 xG for a forward: (2 - 0.6) x 4 - 0.1 x 3.
    assert parts.loc[0, "minutes"] == pytest.approx(1.0)
    assert parts.loc[0, "finishing"] == pytest.approx(1.4 * 4 - 0.3)


def test_decompose_without_xg_or_p_play():
    t = pd.DataFrame({"xp": [3.0, 3.0], "minutes": [0, 90], "total_points": [0, 5], "position": [3, 3]})
    parts = live.decompose(t)
    assert parts["minutes"].tolist() == [-3.0, 0.0]
    assert parts["finishing"].tolist() == [0.0, 0.0]
    assert parts.loc[1, "other"] == pytest.approx(2.0)


def fixture(id_, home, away, gw=6, started=True, done=True, kickoff="2026-10-10T14:00:00Z"):
    return {"id": id_, "event": gw, "team_h": home, "team_a": away, "started": started, "finished": done,
            "finished_provisional": done, "kickoff_time": kickoff, "team_h_score": 1 if started else None,
            "team_a_score": 0 if started else None, "minutes": 90 if done else 30}


def test_club_state():
    states = live.club_state([fixture(1, 1, 2), fixture(2, 3, 4, done=False), fixture(3, 5, 6, started=False, done=False),
                              fixture(4, 1, 7, gw=7, started=False, done=False)], 6)
    assert states == {1: "done", 2: "done", 3: "live", 4: "live", 5: "to play", 6: "to play"}


# A squad: 2 GKP (1, 12), 5 DEF (2-5, 13), 5 MID (6-9, 14), 3 FWD (10, 11, 15). Starting 4-4-2 with 1 GKP.
POSITIONS = {1: 1, 12: 1, 2: 2, 3: 2, 4: 2, 5: 2, 13: 2, 6: 3, 7: 3, 8: 3, 9: 3, 14: 3, 10: 4, 11: 4, 15: 4}
TEAM = {"lineup": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "bench": [12, 13, 14, 15], "captain": 10, "vice": 6,
        "chip": None, "forecast": 50.0, "spread": {"p10": 35, "p50": 49, "p90": 66,
                                                  "histogram": {"start": 30, "width": 10, "shares": [0.2, 0.3, 0.3, 0.2]}}}


def frame(points: dict[int, float], minutes: dict[int, float], state: dict[int, str] | None = None) -> pd.DataFrame:
    ids = list(POSITIONS)
    return pd.DataFrame({"position": [POSITIONS[e] for e in ids], "total_points": [points.get(e, 2.0) for e in ids],
                         "minutes": [minutes.get(e, 90.0) for e in ids], "xp": 3.0,
                         "state": [(state or {}).get(e, "done") for e in ids]}, index=ids)


def test_team_live_counts_captain_and_bench():
    result = live.team_live(TEAM, frame({10: 10.0}, {}))
    assert result["points"] == 10 * 2 + 2 * 10            # captain doubled, ten others on 2, bench not counted
    assert result["settled"] and result["subs"] == []
    assert result["percentile"] == pytest.approx(0.2)    # 40 points: above the 30-40 bin, at the bottom of 40-50


def test_team_live_substitutes_and_vice():
    # The captain (10) and a defender (2) didn't play: the vice takes the armband. Bench order, skipping the
    # keeper (12, didn't play either): 13 on for 2, then 14 on for 10.
    t = frame({6: 7.0, 13: 5.0}, {10: 0.0, 2: 0.0, 12: 0.0})
    result = live.team_live(TEAM, t)
    assert result["armband"] == 6
    assert result["subs"] == [{"out": 2, "in": 13}, {"out": 10, "in": 14}]
    counted = {p["element"] for p in result["picks"] if p["counted"]}
    assert 10 not in counted and 2 not in counted and {13, 14} <= counted


def test_team_live_keeps_a_valid_formation():
    # Three of the four defenders didn't play. The bench keeper can't replace a defender; the forward (15)
    # can replace the first (four at the back become three); after that only a defender keeps three, so 13
    # comes on for the second and the third stays in with 0 points: no one else keeps the formation valid.
    team = dict(TEAM, bench=[12, 15, 14, 13])
    t = frame({}, {2: 0.0, 3: 0.0, 4: 0.0})
    assert live.team_live(team, t)["subs"] == [{"out": 2, "in": 15}, {"out": 3, "in": 13}]


def test_team_live_waits_for_matches_still_to_play():
    t = frame({}, {10: 0.0}, {10: "to play", 11: "live"})
    result = live.team_live(TEAM, t)
    assert result["armband"] == 10 and result["subs"] == []
    assert not result["settled"] and result["still_to_play"] == 2
    assert result["still_xp"] == pytest.approx(3.0 * 2)  # the captain's xP doubled; 11 is already playing
    assert result["percentile"] is None


def test_percentile_from_histogram():
    spread = TEAM["spread"]
    assert live._percentile(spread, 30) == 0
    assert live._percentile(spread, 45) == pytest.approx(0.2 + 0.15)
    assert live._percentile(spread, 99) == pytest.approx(1.0)


def test_due():
    now = datetime(2026, 10, 10, 15, 0, tzinfo=timezone.utc)
    fresh = {"gw": 6, "generated": (now - timedelta(minutes=5)).isoformat()}
    playing = [fixture(1, 1, 2, done=False, kickoff="2026-10-10T14:00:00Z")]
    quiet = [fixture(1, 1, 2, kickoff="2026-10-09T14:00:00Z")]
    assert live.due(None, quiet, 6, now)
    assert live.due({"gw": 5, "generated": fresh["generated"]}, quiet, 6, now)
    assert live.due(fresh, playing, 6, now)                   # a match in play
    assert not live.due(fresh, quiet, 6, now)
    old = {"gw": 6, "generated": (now - timedelta(hours=7)).isoformat()}
    assert live.due(old, quiet, 6, now)


def test_analyse_end_to_end():
    forecast = {"gw": 6, "deadline": "2026-10-10T10:00:00Z", "model": "ensemble", "made_at": None, "players": {
        "element": [1, 2, 3], "team": [1, 2, 3], "position": [4, 3, 2], "xp_6": [4.0, 3.0, 2.0],
        "p_play": [0.9, 0.9, 0.9], "pts_p10": [1, 1, 1], "pts_p90": [10, 8, 6], "p_haul": [0.1, 0.05, 0.02],
        "p_blank": [0.4, 0.5, 0.5], "status": ["d", "a", "a"], "chance": [75, None, None], "selected_by": [30, 10, 1]}}
    live_json = {"elements": [
        {"id": 1, "stats": {"minutes": 90, "total_points": 13, "goals_scored": 2, "assists": 0, "bonus": 3,
                            "expected_goals": "0.1", "expected_assists": "0.0", "starts": 1},
         "explain": [{"fixture": 1, "stats": [{"identifier": "goals_scored", "points": 8, "value": 2}]}]},
        {"id": 2, "stats": {"minutes": 0, "total_points": 0, "starts": 0}},
        {"id": 3, "stats": {"minutes": 90, "total_points": 6, "clean_sheets": 1, "starts": 1, "expected_goals": "0.0"},
         "explain": [{"fixture": 2, "stats": [{"identifier": "clean_sheets", "points": 4, "value": 1}]}]},
    ]}
    fixtures = [fixture(1, 1, 4), fixture(2, 2, 3, done=False)]
    forecast["players"]["status_deadline"] = ["a", "a", "a"]
    forecast["players"]["chance_deadline"] = [100, None, None]
    out = live.analyse(forecast, live_json, fixtures)
    players = pd.DataFrame(out["players"]).set_index("element")
    assert out["done"] == 1 and out["total"] == 2
    assert players.at[1, "state"] == "done" and players.at[3, "state"] == "live"
    assert players.at[1, "news_changed"]                    # doubtful 75% when made, fit at the deadline
    assert players.at[1, "verdict"] == "Finishing"
    assert players.at[1, "band"] == "above"
    assert any("finishing well beyond" in r for r in players.at[1, "reasons"])
    assert out["players"]["verdict"][out["players"]["element"].index(3)] is None   # his match is still on
    assert out["week"]["n"] == 1                            # only finished matches are scored
    assert out["lessons"]["lucky"] == [1] and out["lessons"]["stale"] == [1]
