"""Paths, data sources and tunable settings in one place."""

import os
from pathlib import Path

# Repo root (src/xpfpl/config.py -> repo). Override with XPFPL_HOME if installed elsewhere.
ROOT = Path(os.environ.get("XPFPL_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = ROOT / "models"

BACKTEST_DIR = DATA_DIR / "backtests"

MATCHES_PATH = PROCESSED_DIR / "matches.parquet"
MODEL_PATH = MODELS_DIR / "xp_mlp.pt"              # the default model; the others are xp_<name>.pt
VALIDATION_PATH = MODELS_DIR / "validation.json"   # accuracy report written by `xpfpl train` / `validate`
COMPARISON_PATH = MODELS_DIR / "comparison.json"   # every model on the same season, from `xpfpl compare`
TUNING_PATH = BACKTEST_DIR / "tuning.json"         # the tuning report written by `xpfpl tune`

FPL_API = "https://fantasy.premierleague.com/api"
VAASTAV_RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# Completed seasons pulled from the vaastav repo. The current season always comes from the FPL API.
HISTORY_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# Squad rules (also available from bootstrap-static game_settings / element_types).
SQUAD_SIZE = {1: 2, 2: 5, 3: 5, 4: 3}
LINEUP_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
LINEUP_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
MAX_PER_CLUB = 3
MAX_FREE_TRANSFERS = 5
HIT_COST = 4

# Which model `predict`, `recommend` and the dashboard use by default. See xpfpl.models.NAMES.
MODEL = "mlp"

# Optimiser tuning parameters, all set by `xpfpl tune`: it scores candidate values by replaying 2023-24 and
# 2024-25 in full and keeps whatever wins. The report is data/backtests/tuning.json.
HORIZON = 3                # gameweeks to look ahead
DISCOUNT = 0.8             # weight of GW t+k relative to GW t is DISCOUNT**k (later predictions are less certain)
BENCH_WEIGHT = 0.05        # value of bench points (they only count via auto-subs)
FT_VALUE = 3.0             # xP value of rolling a free transfer to next week
MAX_HITS = 0               # extra transfers beyond the free allowance, at -4 points each
PLAN_TRANSFERS = False     # plan a squad per gameweek across the horizon, not one squad held throughout
PRICE_WEIGHT = 1.0         # xP per £m of expected price change (0 = ignore price rises entirely)
POOL_SIZE = 50             # candidates per position handed to the optimiser
# Betting-market news: a player whose anytime-goalscorer odds are at or below markets.OUT_THRESHOLD
# (6%) at prediction time has almost certainly been ruled out. On 2025-26 those players scored
# 0.06 points against the model's 0.61, so the next gameweek's xP is multiplied by this.
MARKET_OUT_XP_FACTOR = 0.1

# Chip thresholds (xP gained vs. not playing the chip). These four are still the original
# guesses: the chip stages of `xpfpl tune` have not been run to completion yet, so treat the
# chip advice as weaker than the rest. `xpfpl tune --stages wildcard` tunes one of them.
TRIPLE_CAPTAIN_MIN_XP = 9.0
BENCH_BOOST_MIN_XP = 12.0
FREE_HIT_MIN_GAIN = 12.0
WILDCARD_MIN_GAIN = 20.0
