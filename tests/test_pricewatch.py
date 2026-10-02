"""The price tracker logs moves between runs, keeps a rolling progress history and a daily snapshot."""

import pandas as pd
import pytest

from xpfpl.data import pricewatch


def bootstrap(costs: dict[int, int], pct: dict[int, float] | None = None, change_event: dict[int, int] | None = None) -> dict:
    pct = pct or {}
    change_event = change_event or {}
    return {
        "events": [{"id": 1, "deadline_time": "2099-08-15T17:30:00Z", "is_current": True, "is_next": False},
                   {"id": 2, "deadline_time": "2099-08-22T17:30:00Z", "is_current": False, "is_next": True}],
        "teams": [{"id": 1, "code": 3}],
        "elements": [{"id": i, "code": 100 + i, "web_name": f"P{i}", "team": 1, "element_type": 3, "now_cost": c,
                      "cost_change_event": change_event.get(i, 0), "cost_change_start": 0, "selected_by_percent": "5.0",
                      "transfers_in_event": 10, "transfers_out_event": 3, "status": "a",
                      "price_change_percent": str(pct.get(i, 0.0)), "price_change_hourly_rate": 1,
                      "price_change_projections": [{"offset": 0, "projected_percent": "50.0", "likelihood": 2}],
                      "price_change_locked_until": None} for i, c in costs.items()],
    }


T0 = pd.Timestamp("2099-08-16T10:00Z")


def test_first_run_lists_this_gameweeks_moves_without_a_time():
    state, daily = pricewatch.update(None, bootstrap({1: 50, 2: 61}, change_event={2: 1}), now=T0)
    assert [(c["id"], c["from"], c["to"], c["t"]) for c in state["changes"]] == [(2, 60, 61, None)]
    assert state["season"] == "2099-00" and state["gw_next"] == 2
    assert daily is None                                    # before DAILY_HOUR_UTC


def test_later_runs_log_moves_and_keep_three_days_of_progress():
    state, _ = pricewatch.update(None, bootstrap({1: 50, 2: 61}, pct={1: 40.0}), now=T0)
    state, _ = pricewatch.update(state, bootstrap({1: 51, 2: 61}, pct={1: 2.0}), now=T0 + pd.Timedelta(hours=1))
    moved = [c for c in state["changes"] if c["t"]]
    assert [(c["id"], c["from"], c["to"]) for c in moved] == [(1, 50, 51)]
    p1 = next(p for p in state["players"] if p["id"] == 1)
    assert p1["h"] == [40.0, 2.0] and len(state["times"]) == 2
    later = T0 + pd.Timedelta(hours=pricewatch.HISTORY_HOURS + 2)
    state, _ = pricewatch.update(state, bootstrap({1: 51, 2: 61}, pct={1: 7.0}), now=later)
    assert next(p for p in state["players"] if p["id"] == 1)["h"] == [7.0]   # older runs dropped
    assert len(state["changes"]) == 1                                         # the log is kept


def test_one_daily_snapshot_a_day():
    evening = T0.normalize() + pd.Timedelta(hours=pricewatch.DAILY_HOUR_UTC, minutes=7)
    state, daily = pricewatch.update(None, bootstrap({1: 50}), now=evening)
    assert daily is not None and len(daily) == 1
    _, again = pricewatch.update(state, bootstrap({1: 50}), now=evening + pd.Timedelta(hours=1))
    assert again is None


def test_a_new_season_starts_a_new_log():
    old = {"season": "2098-99", "players": [{"id": 1, "cost": 40, "h": [1.0]}], "changes": [{"id": 9}], "times": ["x"]}
    state, _ = pricewatch.update(old, bootstrap({1: 50}), now=T0)
    assert state["changes"] == [] and state["times"] == [T0.isoformat()]


def test_change_log_sorts_newest_first():
    log = pricewatch.change_log({"changes": [
        {"t": None, "gw": 1, "id": 1, "code": 1, "name": "a", "team_code": 3, "pos": 3, "from": 50, "to": 51},
        {"t": "2099-08-20T01:07:00+00:00", "gw": 2, "id": 2, "code": 2, "name": "b", "team_code": 3, "pos": 3, "from": 50, "to": 49},
        {"t": "2099-08-21T01:07:00+00:00", "gw": 2, "id": 3, "code": 3, "name": "c", "team_code": 3, "pos": 3, "from": 50, "to": 51}]})
    assert log["id"].tolist() == [3, 2, 1]
