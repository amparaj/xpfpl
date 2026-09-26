import json

import pandas as pd

from xpfpl import config, modelteam


def _week(gw, chip=None, squad=None, bank=1.0, free=1, lineup=None, bench=None, captain=1, vice=2):
    squad = squad or list(range(1, 16))
    return {"season": "2026-27", "gw": gw, "source": "live", "model": "mlp", "made_at": "", "chip": chip,
            "free_transfers": free, "bank": bank, "hits": 0, "transfers": [], "squad": squad,
            "lineup": lineup or squad[:11], "bench": bench or squad[11:], "captain": captain, "vice": vice, "xp": 50.0,
            "next": {"squad": {str(p): 50 for p in squad}, "bank": bank, "free_transfers": free}}


def _save(tmp_path, monkeypatch, weeks):
    monkeypatch.setattr(modelteam, "FOLDER", tmp_path)
    for w in weeks:
        path = tmp_path / "2026-27" / f"gw{w['gw']:02d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(w), encoding="utf-8")


def test_state_starts_from_scratch(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, [])
    assert modelteam._state(modelteam.decisions("2026-27"), 1) == ({}, 100.0, 1, [])


def test_state_carries_the_last_decision_and_banks_missed_weeks(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, [_week(1), _week(2, chip="3xc", bank=2.5, free=2)])
    squad, bank, free, used = modelteam._state(modelteam.decisions("2026-27"), 3)
    assert (len(squad), bank, free) == (15, 2.5, 2)
    assert used == [{"name": "3xc", "event": 2}]
    # GW3 and GW4 had no decision: each banks a free transfer, up to the cap.
    _, _, free, _ = modelteam._state(modelteam.decisions("2026-27"), 5)
    assert free == 4
    _, _, free, _ = modelteam._state(modelteam.decisions("2026-27"), 9)
    assert free == config.MAX_FREE_TRANSFERS


def test_season_record_scores_autosubs_and_carries_a_missed_week(tmp_path, monkeypatch):
    # GKP 1 and 15, DEF 2-6, MID 7-11, FWD 12-14; a 4-4-2 with DEF 6, MID 11 and FWD 14 on the bench.
    squad = list(range(1, 16))
    position = {1: 1, 15: 1, **{p: 2 for p in range(2, 7)}, **{p: 3 for p in range(7, 12)}, **{p: 4 for p in range(12, 15)}}
    lineup = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]          # 4-4-2
    bench = [15, 6, 11, 14]
    _save(tmp_path, monkeypatch, [_week(1, lineup=lineup, bench=bench, captain=12, vice=13)])
    bs = {"elements": [{"id": p, "web_name": f"P{p}", "team": p, "element_type": position[p], "now_cost": 50}
                       for p in squad]}
    rows = pd.DataFrame([{"round": gw, "element": p, "total_points": 2, "minutes": 90} for gw in (1, 2) for p in squad])
    rows.loc[(rows["round"] == 1) & (rows["element"] == 2), ["total_points", "minutes"]] = 0   # a defender doesn't play
    record = modelteam.season_record("2026-27", rows, bs)
    week1, week2 = record["gameweeks"]
    assert week1["autosubs"] == [[2, 6]]                    # the first outfield sub who keeps the formation
    assert week1["points"] == 11 * 2 + 2                    # eleven players on 2, the captain's 2 doubled
    assert week2["source"] == "carried" and week2["points"] == 24
    assert record["next"] is None


def test_team_forecast_counts_what_the_week_scores():
    base = {"lineup": [1, 2, 3], "bench": [4, 5], "captain": 2, "chip": None}
    xp = pd.Series({1: 2.0, 2: 5.0, 3: 1.0, 4: 3.0, 5: 0.5})
    assert modelteam.team_forecast({**base, "xp": 13.0}) == (13.0, "decision")
    assert modelteam.team_forecast({**base, "xp": 13.0, "chip": "bboost", "bench_xp": 3.5}) == (16.5, "decision")
    # an older decision without bench_xp takes its Bench Boost bench from the week's xP
    assert modelteam.team_forecast({**base, "xp": 13.0, "chip": "bboost"}, xp) == (16.5, "decision")
    # a carried-over week: summed from the per-player xP, captain doubled (tripled with Triple Captain)
    assert modelteam.team_forecast({**base, "xp": None}, xp) == (13.0, "gameweek xP")
    assert modelteam.team_forecast({**base, "xp": None, "chip": "3xc"}, xp) == (18.0, "gameweek xP")
    assert modelteam.team_forecast({**base, "xp": None}) == (None, None)
