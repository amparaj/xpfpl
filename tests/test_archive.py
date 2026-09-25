import json

import pandas as pd
import pytest

from xpfpl.data import archive, history


@pytest.fixture
def tmp_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "FPL", tmp_path / "fpl")
    monkeypatch.setattr(archive, "POLYMARKET", tmp_path / "polymarket")
    monkeypatch.setattr(archive, "PREDICTIONS", tmp_path / "predictions")
    monkeypatch.setattr(archive, "_PRICES", None)
    return tmp_path


def test_frame_makes_nested_and_mixed_columns_storable(tmp_path):
    df = archive.frame([{"a": 1, "nested": {"x": 1}, "mixed": 1}, {"a": 2, "nested": [1, 2], "mixed": "two"}])
    assert df["nested"].tolist() == ['{"x": 1}', "[1, 2]"]
    assert df["mixed"].tolist() == ["1", "two"]
    assert archive.write(df, tmp_path / "t.parquet")


def test_write_skips_an_unchanged_table(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "t.parquet"
    assert archive.write(df, path)
    assert not archive.write(df.copy(), path)
    assert archive.write(df.assign(a=[1, 3]), path)


def _bootstrap(next_deadline="2099-01-01T10:00:00Z"):
    return {
        "events": [{"id": 1, "deadline_time": "2026-08-15T10:00:00Z", "finished": True, "data_checked": True,
                    "is_next": False, "chip_plays": [{"chip_name": "wildcard", "num_played": 5}]},
                   {"id": 2, "deadline_time": next_deadline, "finished": False, "data_checked": False,
                    "is_next": True, "chip_plays": []}],
        "teams": [{"id": 1, "code": 3, "name": "Arsenal", "short_name": "ARS"},
                  {"id": 2, "code": 8, "name": "Chelsea", "short_name": "CHE"}],
        "elements": [{"id": 10, "code": 100, "first_name": "A", "second_name": "Home", "web_name": "Home",
                      "element_type": 3, "team": 1, "team_code": 3, "news": "", "now_cost": 55},
                     {"id": 20, "code": 200, "first_name": "B", "second_name": "Away", "web_name": "Away",
                      "element_type": 4, "team": 2, "team_code": 8, "news": "Knock", "now_cost": 70}],
    }


def _history_row(element, home, gw=1, fixture=1, points=2):
    return {"element": element, "fixture": fixture, "opponent_team": 2 if home else 1, "total_points": points,
            "was_home": home, "kickoff_time": "2026-08-16T14:00:00Z", "team_h_score": 1, "team_a_score": 0,
            "round": gw, "minutes": 90, "goals_scored": int(home), "value": 55}


def test_api_season_round_trips_through_the_past_season_loader(tmp_archive):
    bs = _bootstrap()
    fx = [{"id": 1, "event": 1, "team_h": 1, "team_a": 2, "finished": True, "stats": [{"a": 1}]},
          {"id": 2, "event": 2, "team_h": 2, "team_a": 1, "finished": False, "stats": []}]
    rows = [_history_row(10, True), _history_row(20, False),
            _history_row(10, False, gw=2, fixture=2)]            # GW2 isn't checked yet: not archived
    assert archive.save_api_season("2026-27", bs, fx, rows) > 0
    assert archive.save_api_season("2026-27", bs, fx, rows) == 0        # nothing changed
    assert archive.has_season("2026-27")
    assert [p.name for p in (tmp_archive / "fpl" / "2026-27" / "gws").iterdir()] == ["gw01.parquet"]

    season = history.load_past_season("2026-27")
    assert len(season) == 2
    home = season[season["element"] == 10].iloc[0]
    assert (home["team_code"], home["opp_code"], home["name"], home["position"]) == (3, 8, "Home", 3)

    # A player who drops out of bootstrap-static stays in the archive's who's who.
    bs["elements"] = bs["elements"][:1]
    archive.save_api_season("2026-27", bs, fx, rows[:1])
    assert set(archive.season_tables("2026-27")[1]["id"]) == {10, 20}


def test_deadline_snapshot_only_before_the_deadline(tmp_archive):
    path = archive.save_deadline(_bootstrap())
    assert path.name == "gw02.parquet"
    assert archive.deadline_snapshots("2026-27")[2]["news"].tolist() == ["", "Knock"]
    assert archive.save_deadline(_bootstrap(next_deadline="2000-01-01T00:00:00Z")) is None


def test_polymarket_events_and_prices_come_back(tmp_archive, tmp_path, monkeypatch):
    from xpfpl.data import markets
    raw = tmp_path / "raw"
    monkeypatch.setattr(markets, "RAW_DIR", raw)
    (raw / "history_14d").mkdir(parents=True)
    (raw / "history_14d" / "tok1.json").write_text(json.dumps([{"t": 1, "p": 0.4}, {"t": 2, "p": 0.5}]))
    (raw / "history_14d" / "tok2.json").write_text("[]")
    event = {"slug": "epl-ars-che-2026-08-16", "startTime": "2026-08-16T14:00:00Z", "description": "long text",
             "teams": [{"name": "Arsenal FC", "ordering": "home"}, {"name": "Chelsea FC", "ordering": "away"}],
             "markets": [{"question": "Will Arsenal FC win on 2026-08-16?", "clobTokenIds": '["tok1", "no1"]',
                          "description": "rules"},
                         {"question": "Will Arsenal FC vs. Chelsea FC end in a draw?", "clobTokenIds": '["tok2", "no2"]'}]}
    games = {event["slug"]: [event]}
    found = pd.DataFrame([{"slug": event["slug"], "season": "2026-27", "gw": 1, "fixture": 1}])
    assert archive.save_polymarket(games, found) == 2

    back = archive.polymarket_games()[event["slug"]][0]
    assert "description" not in back and "description" not in back["markets"][0]
    assert archive.price_history("history_14d", "tok1") == [{"t": 1, "p": 0.4}, {"t": 2, "p": 0.5}]
    assert archive.price_history("history_14d", "tok2") == []           # archived as "no trades"
    assert archive.price_history("history_14d", "tok3") is None
