import numpy as np
import pandas as pd
import pytest

from xpfpl import prices


def make_history(n_players: int = 60, gws: int = 20, seed: int = 0) -> pd.DataFrame:
    """A season where in-form players drift up in price and out-of-form ones drift down."""
    rng = np.random.default_rng(seed)
    rows = []
    for element in range(1, n_players + 1):
        form = rng.uniform(0, 9)          # points per match this player averages
        value = 50 + int(form * 5)
        for gw in range(1, gws + 1):
            rows.append({"season": "2099-00", "element": element, "gw": gw,
                         "kickoff_time": pd.Timestamp("2099-08-01", tz="UTC") + pd.Timedelta(days=7 * gw),
                         "value": value, "price": value / 10.0,
                         "total_points_r3": form + rng.normal(0, 0.3)})
            value += int(rng.random() < (form - 4) / 10)      # good form -> rises
            value -= int(rng.random() < (4 - form) / 10)      # poor form -> falls
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def history():
    return make_history()


def test_fitted_table_rewards_form(history):
    table = prices.fit(history, min_cell=20)   # a real fit has 250k rows; this one has 1,200
    cells = np.array(table["cells"])
    assert cells.shape == (len(prices.FORM_BINS) - 1, len(prices.PRICE_BINS) - 1)
    # The best form band must drift up relative to the worst.
    assert cells[-1].mean() > cells[0].mean()


def test_expected_delta_is_in_millions_per_gameweek(history):
    table = prices.fit(history, min_cell=20)
    players = history.drop_duplicates("element").set_index("element")
    delta = prices.expected_delta(players, table)

    assert len(delta) == len(players)
    assert delta.abs().max() < 0.2      # a tenth or two a week, never pounds
    in_form = players["total_points_r3"] > 7
    if in_form.any():
        assert delta[in_form].mean() > delta[~in_form].mean()


def test_table_is_cached_and_can_be_limited_to_earlier_seasons(history, tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "PATH", tmp_path / "prices.json")
    first = prices.table(history)
    assert prices.PATH.exists()
    assert prices.table() == first          # second call reads the cache, no frame needed

    older = history.assign(season="2098-99")
    combined = pd.concat([older, history])
    limited = prices.fit(combined, before_season="2099-00")
    assert limited["rows"] < prices.fit(combined)["rows"]    # the later season was excluded


def test_live_momentum_follows_net_transfers():
    elements = pd.DataFrame([
        {"id": 1, "selected_by_percent": "10.0", "transfers_in_event": 200_000, "transfers_out_event": 1_000},
        {"id": 2, "selected_by_percent": "10.0", "transfers_in_event": 1_000, "transfers_out_event": 200_000},
        {"id": 3, "selected_by_percent": "10.0", "transfers_in_event": 5_000, "transfers_out_event": 5_000},
    ]).set_index("id")
    delta = prices.momentum_delta(elements)

    assert delta[1] > 0 and delta[2] < 0
    assert delta[3] == pytest.approx(0.0)
    assert delta.abs().max() <= 0.1
