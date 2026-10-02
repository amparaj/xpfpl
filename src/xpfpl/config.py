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
ARCHIVE_DIR = ROOT / "archive"                     # git-tracked copy of every source (data/archive.py)
SITE_DATA_DIR = ROOT / "web" / "public" / "data"   # what `xpfpl export` writes for the website

BACKTEST_DIR = DATA_DIR / "backtests"
MY_TEAM_DIR = DATA_DIR / "myteam"                  # the teams you save in the dashboard's Plan Ahead
APP_SETTINGS_PATH = DATA_DIR / "app_settings.json"  # the dashboard's settings (your team id)

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
MODEL = "ensemble"

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
# The component model's clean-sheet and goals-conceded heads, moved this far towards the match
# odds where there are any (models/components.py `_blend_market`): out of sample RMSE -0.0011
# on 2024-25 and -0.0041 on 2025-26 at 0.5, against 0. Goals and assists stay the model's own.
MARKET_DEFENCE_WEIGHT = 0.5
# Penalty takers (setpieces.py): weight on the change in a player's penalty role (FPL's
# penalties_order now, against the penalties he took lately). 0 = recorded in the forecast
# (`pen_order`, `pen_xp`) but not applied: on 2025-26 its best weight was 0.29 (se 0.28).
PENALTY_WEIGHT = 0.0
# Monte Carlo (simulate.py): how many times `predict` plays each upcoming gameweek to give every
# xP a range and the chances behind it. It only describes risk: the optimiser still maximises xP.
# 0 switches it off.
SIM_RUNS = 5000
# Confidence in a player's next-gameweek xP (predict.py): how far the ensemble's members disagree,
# as their standard deviation over the xP (floored at 1). At or below the first cut "High", above
# the second "Low". Fixed 2026-10-02 on the robustness run's held-out 2020-21..2022-23 forecasts
# (players on 2+ xP: half High, 15% Low); on 2023-24..2025-26 the Low tier's squared misses were
# 1.19x the usual for the same xP, High's 0.97x. The members' spread beat the chance of playing,
# the player's history and seed-to-seed spread as a signal.
CONFIDENCE_CUTS = (0.09, 0.164)

# Chip thresholds (xP gained vs. not playing the chip), tuned 2026-10-02 with the ensemble on top
# of the settings above (`xpfpl tune` chip stages, 2020-21 to 2023-24, 4 replays each; report in
# data/backtests/tuning_2026-10-02_chips.json). Were 9 / 12 / 12 / 20 (guesses). In-sample +30
# points a season, mostly the wildcard; on the unseen 2024-25 and 2025-26 +16 and +5 (2252 vs 2241
# a season, ± ~30): better in both but within the noise, so the chip advice stays the weakest part.
TRIPLE_CAPTAIN_MIN_XP = 6.0
BENCH_BOOST_MIN_XP = 8.0
FREE_HIT_MIN_GAIN = 18.0
WILDCARD_MIN_GAIN = 30.0
# Warn once the gameweeks left in a chip window are no more than the unused chips plus this many
# spare weeks (one chip per GW, so three chips and six weeks left gives three weeks of slack).
CHIP_SPARE_WEEKS = 3
