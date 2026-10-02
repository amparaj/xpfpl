"""Bookmaker odds (Football-Data.co.uk): column choice, margin removal and fixture mapping."""

import numpy as np
import pandas as pd
import pytest

from xpfpl.data import archive, bookmakers


def _raw(**odds) -> pd.DataFrame:
    base = {"Date": ["21/08/2026", "22/08/2026"], "Time": ["20:00", "12:30"],
            "HomeTeam": ["Arsenal", "Hull"], "AwayTeam": ["Coventry", "Man United"],
            "FTHG": [3, 2], "FTAG": [0, 0]}
    return pd.DataFrame({**base, **odds})


def _table(tmp_path, monkeypatch, raw: pd.DataFrame) -> pd.DataFrame:
    monkeypatch.setattr(bookmakers, "ARCHIVE", tmp_path)
    archive.write(raw, tmp_path / "2026-27.parquet")
    return bookmakers.table("2026-27")


def test_odds_become_fair_probabilities(tmp_path, monkeypatch):
    t = _table(tmp_path, monkeypatch, _raw(AvgH=[1.2, 8.4], AvgD=[6.8, 4.9], AvgA=[14.2, 1.36],
                                           **{"Avg>2.5": [1.55, 1.76], "Avg<2.5": [2.38, 1.98]}))
    assert np.allclose(t[["home_win", "draw", "away_win"]].sum(axis=1), 1)
    assert t.at[0, "home_win"] > 0.75 and t.at[1, "away_win"] > 0.65
    assert 0.5 < t.at[1, "over_2.5"] < 0.55          # 1.76 / 1.98 with the margin taken out
    assert list(t["home_code"]) == [3, 88] and list(t["away_code"]) == [9, 1]
    assert t.at[0, "kickoff"] == pd.Timestamp("2026-08-21 19:00", tz="UTC")    # BST to UTC


def test_older_column_names_and_bet365_fallback(tmp_path, monkeypatch):
    t = _table(tmp_path, monkeypatch, _raw(BbAvH=[1.2, np.nan], BbAvD=[6.8, np.nan], BbAvA=[14.2, np.nan],
                                           B365H=[1.25, 8.0], B365D=[6.5, 5.0], B365A=[13.0, 1.4]))
    assert t["home_win"].notna().all()
    assert t.at[1, "away_win"] > 0.6 and t["over_2.5"].isna().all()


def test_an_unknown_club_is_an_error(tmp_path, monkeypatch):
    raw = _raw(AvgH=[2, 2], AvgD=[3, 3], AvgA=[4, 4]).assign(HomeTeam=["Arsenal", "Wrexham"])
    with pytest.raises(KeyError, match="Wrexham"):
        _table(tmp_path, monkeypatch, raw)
