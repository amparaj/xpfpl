"""Which players are about to rise or fall in price, and what that is worth.

FPL prices move with net transfers: a player everyone is buying goes up £0.1m, one everyone is
selling goes down. Nobody outside FPL knows the exact threshold, but the *effect* is easy to
measure from the price history we already have, and it is real:

    points per match over the last 3        average price change per gameweek
    0 to 1                                  -0.004m
    3 to 5                                  +0.001m
    5 to 7                                  +0.013m
    7+                                      +0.032m

So form predicts price. `table()` fits that lookup (form band x price band -> average change in
tenths of a million per gameweek) from completed seasons and caches it in models/prices.json;
`expected_delta()` applies it to a set of players.

The optimiser then treats team value as worth `config.PRICE_WEIGHT` points per £m: buying a
player just before he rises means a cheaper squad later, which eventually buys better players.
The weight is small on purpose - price is a tie-breaker between similar options, never a reason
to pick a worse player - and it is tuned with the backtest like every other parameter.

Live, `momentum_delta()` refines this with the one thing the API tells us and history doesn't:
how many managers transferred each player in or out this week.
"""

import json

import numpy as np
import pandas as pd

from xpfpl import config

FORM_BINS = [-1, 0.5, 1.5, 3, 5, 7, 1000]           # points per match over the last 3 matches
PRICE_BINS = [0, 5, 7, 9, 100]                      # £m
FORM_COL = "total_points_r3"
PATH = config.MODELS_DIR / "prices.json"

# Net transfers in a gameweek, as a share of a player's owners, that maps to a full 0.1m move.
# A rough stand-in for FPL's hidden threshold, used only for the live momentum adjustment.
MOMENTUM_SCALE = 0.25
MIN_CELL = 200          # rows a (form, price) cell needs before it beats the overall average


def _bucket(values: pd.Series, bins: list) -> np.ndarray:
    return np.clip(np.digitize(values.fillna(0).to_numpy(dtype="float64"), bins[1:-1]),
                   0, len(bins) - 2)


def fit(frame: pd.DataFrame, before_season: str | None = None, min_cell: int = MIN_CELL) -> dict:
    """Average price change per gameweek, by form band and price band.

    `before_season` restricts the fit to earlier seasons, which is what the backtest needs:
    the manager replaying 2022-23 could not have measured 2024-25's price moves.
    """
    df = frame
    if before_season:
        df = df[df["season"].str[:4].astype(int) < int(before_season[:4])]
    df = df.sort_values(["season", "element", "kickoff_time"])
    delta = df.groupby(["season", "element"])["value"].shift(-1) - df["value"]
    ok = delta.notna()
    form, price = _bucket(df[FORM_COL][ok], FORM_BINS), _bucket(df["price"][ok], PRICE_BINS)

    grid = pd.DataFrame({"form": form, "price": price, "delta": delta[ok].to_numpy()})
    means = grid.groupby(["form", "price"])["delta"].mean()
    counts = grid.groupby(["form", "price"])["delta"].size()
    overall = float(grid["delta"].mean())
    cells = [[float(means.get((f, p), overall)) if counts.get((f, p), 0) >= min_cell else overall
              for p in range(len(PRICE_BINS) - 1)] for f in range(len(FORM_BINS) - 1)]
    return {"cells": cells, "overall": overall, "rows": int(ok.sum()),
            "seasons": f"up to {df['season'].max()}" if len(df) else ""}


def table(frame: pd.DataFrame | None = None, before_season: str | None = None,
          cache: bool = True) -> dict:
    """The fitted table: from models/prices.json if it is there, otherwise fitted and saved."""
    if before_season is None and cache and PATH.exists():
        try:
            return json.loads(PATH.read_text(encoding="utf-8"))
        except ValueError:
            pass
    if frame is None:
        from xpfpl.data.history import load_matches
        from xpfpl.features import build_training_frame
        frame = build_training_frame(load_matches())
    fitted = fit(frame, before_season)
    if before_season is None and cache:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(fitted, indent=1), encoding="utf-8")
    return fitted


def expected_delta(players: pd.DataFrame, fitted: dict | None = None) -> pd.Series:
    """Expected price change in £m per gameweek, for a frame with `total_points_r3` and `price`."""
    fitted = fitted or table()
    cells = np.array(fitted["cells"], dtype="float64")
    form = _bucket(players[FORM_COL], FORM_BINS) if FORM_COL in players else np.zeros(len(players), int)
    price = _bucket(players["price"], PRICE_BINS)
    return pd.Series(cells[form, price] / 10.0, index=players.index)   # tenths -> £m


def momentum_delta(elements: pd.DataFrame) -> pd.Series:
    """Live adjustment from this week's net transfers, in £m per gameweek.

    Net transfers are measured against the number of managers who own the player, so a player
    with 1% ownership being bought hard moves faster than a 40%-owned one being nudged.
    """
    owners = pd.to_numeric(elements["selected_by_percent"], errors="coerce").fillna(0.1).clip(lower=0.05)
    net = (pd.to_numeric(elements["transfers_in_event"], errors="coerce").fillna(0)
           - pd.to_numeric(elements["transfers_out_event"], errors="coerce").fillna(0))
    share = net / (owners / 100.0 * max(net.abs().sum(), 1.0) + 1.0)
    return (np.tanh(share / MOMENTUM_SCALE) / 10.0).clip(-0.1, 0.1)


def for_upcoming(frame: pd.DataFrame, elements: pd.DataFrame | None = None,
                 fitted: dict | None = None) -> pd.Series:
    """Expected £m per gameweek per element id: the form table, plus live momentum if available.

    `frame` is one row per player per upcoming fixture (features.build_future_frame), so it is
    collapsed to one row per player first.
    """
    per_player = frame.drop_duplicates("element").set_index("element")
    delta = expected_delta(per_player, fitted)
    if elements is not None and "transfers_in_event" in elements:
        live = momentum_delta(elements.set_index("id") if "id" in elements else elements)
        delta = (delta + live.reindex(delta.index).fillna(0.0)) / 2.0
    return delta
