from xpfpl.data import api


def _bs(finished: bool, data_checked: bool) -> dict:
    return {"events": [
        {"id": 1, "is_current": False, "finished": True, "data_checked": True},
        {"id": 2, "is_current": True, "finished": finished, "data_checked": data_checked},
        {"id": 3, "is_current": False, "finished": False, "data_checked": False},
    ]}


FX = [
    {"event": 2, "kickoff_time": "2026-09-19T14:00:00Z", "finished": True, "finished_provisional": True},
    {"event": 2, "kickoff_time": "2026-09-21T19:00:00Z", "finished": False, "finished_provisional": False},
    {"event": 3, "kickoff_time": "2026-09-26T14:00:00Z", "finished": False, "finished_provisional": False},
]


def test_gameweek_status_phases():
    live = api.gameweek_status(_bs(False, False), FX)
    assert (live["gw"], live["phase"], live["played"], live["matches"]) == (2, "live", 1, 2)
    assert live["last_kickoff"].isoformat() == "2026-09-21T19:00:00+00:00"
    assert api.gameweek_status(_bs(True, False), FX)["phase"] == "checking"
    assert api.gameweek_status(_bs(True, True), FX)["phase"] == "complete"


def test_gameweek_status_pre_season():
    bs = {"events": [{"id": 1, "is_current": False, "finished": False, "data_checked": False}]}
    assert api.gameweek_status(bs, FX)["phase"] == "pre-season"
