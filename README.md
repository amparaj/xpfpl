# xP-FPL: Expected Points for FPL

A machine learning sandbox predicting expected points (xP) and optimizing squad selection. Built for data analysis and learning.

Each gameweek it tells you:

- **Team selection**: formation, starting XI, bench order, captain and vice-captain
- **Transfers**: whether to use (or roll) your free transfers, and whether a -4 hit is worth it
- **Chips**: whether this is the week for Wildcard, Free Hit, Triple Captain or Bench Boost

## How it works

1. **Data** (`xpfpl/data/`): past seasons (2016-17 onwards) come from the
   [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) repo, and the
   current season comes live from the FPL API. All of it is kept in [`archive/`](archive/README.md),
   compressed in git, so the history survives if a source disappears. Both become one table of player-match rows,
   `data/processed/matches.parquet`. Players and clubs are linked across seasons with their
   persistent `code`s, because FPL re-numbers ids every season.
2. **Features** (`xpfpl/features.py`): rolling 3/5/10-match averages of points, minutes, goals,
   assists, bonus, ICT, xG/xA, defensive contributions and so on; per-90 rates over the last 20
   and 38 matches; points at the same venue; club goals and xG for/against; each fixture's
   expected goals from a fitted team attack/defence model (`xpfpl/teams.py`); the crowd's net
   transfers before the deadline and ownership; and where a player ranks at his club by price and
   minutes; and the betting market's view of each match (expected goals, clean-sheet and win
   chances, from Polymarket's odds at the deadline). Every feature only uses information from
   *before* the match, so there's no leakage.
3. **Prediction** (`xpfpl/models/`): seven interchangeable models predict points for each fixture
   (see [Models](#models)), all compared against a no-ML baseline (5-match average) that they have
   to beat to be worth using. Double gameweeks sum their fixtures and FPL's injury flags scale the xP.
4. **Decisions** (`xpfpl/optimise.py`, `xpfpl/chips.py`): an integer linear program picks a 15-man
   squad, XI and captain for *every* gameweek in the horizon, linked by transfers, to maximise
   discounted xP under FPL's rules (budget, 2/5/5/3, max 3 per club, valid formations, free
   transfers banked up to 5, -4 hits). Chips are simple rules: the xP gained by playing a chip
  must pass a threshold. Every tuning parameter in `config.py` is set by `xpfpl tune` (see [Tuning](#tuning)).

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build if you have a GPU
pip install -e ".[dev]"
```

## Dashboard

```bash
pip install -e ".[app]"   # once: Streamlit + Altair
xpfpl app                 # opens http://localhost:8501
```

Everything in one place, no command line needed. The tabs follow the Guide's weekly routine
(look back at the week, research, plan), then the season as a whole:

- **Guide**: how it works, the weekly routine, how accurate each model is, what the season replays
  scored, how the tuning parameters were chosen, and a glossary.
- **Gameweek Review**: any finished GW. Your points against the average, what the model expected
  per player, the best XI you could have picked from the same squad, and that week's top scorers.
- **Players & Fixtures**: xP for every player with filters, and a fixture difficulty ticker.
- **Markets**: what the betting markets think ("Market Odds") next to the model's own team
  ratings ("Our Odds"). For the next gameweek: win/draw/loss odds, each side's expected goals and
  clean-sheet chance, how the odds moved, the biggest moves over the 90 minutes or 1, 3, 7 or 14 days before the deadline (team news
  arriving in the market) and anytime-goalscorer odds. For a played gameweek, each of those is set
  against what happened: the result, each side's actual xG and goals, clean sheets kept, who
  scored, and which forecast was closer. Plus the season-long markets (title, top four,
  relegation, top scorer, most assists, most clean sheets), now against the start of the season.
- **Plan Ahead**: transfers, XI, bench, captain and chip advice for the next gameweek, and the team
  for each later GW in the horizon. Set an active chip (Wildcard, Free Hit, Triple Captain, Bench
  Boost), override free transfers, bank and hits, and force or ban players.
- **My Season**: points and rank by gameweek, chips used and left, transfer history.

The top of the sidebar counts down to the next deadline and shows where the current gameweek is:
**in progress** (matches played so far), **finished** (FPL still confirming bonus points and
corrections, usually a few hours: don't fetch yet, since fetched data is cached) or **complete**
(ready to fetch). Below it are the data buttons in the order to run them, with warnings when the
data or the model is out of date:

| When | Button | Command | What it does |
| --- | --- | --- | --- |
| Any time before a GW deadline | Refresh live FPL data | - | Re-downloads today's prices, injury flags, fixtures and your team (otherwise cached for 5-10 min). Saves nothing and needs no retraining |
| Start of the season / after a GW is complete | 1. Fetch match data | `xpfpl fetch` | Rebuilds the saved training dataset (`matches.parquet`) from every finished match |
| | 2. Fetch betting odds | `xpfpl markets` | Polymarket odds for every match since 2024-25, saved to disk |
| | 3. Retrain | `xpfpl train` | Refits the model on the two above. Run it last |
| | 4. Publish the website | `xpfpl publish` | Exports the season and pushes the [public site](#website) to GitHub Pages |

Fetch and Refresh both call the FPL API. Fetch is the weekly rebuild of the history the model
learns from. Refresh is a quick reload of what changes day to day (prices, flags, team news) for
the pages you're looking at.

Your team id is remembered in `data/app_settings.json` (gitignored).

## Usage

```bash
xpfpl fetch                        # download history + this season (re-run after each gameweek)
xpfpl markets                      # betting odds (Polymarket) for every match since 2024-25
xpfpl train                        # train the model (after fetch and markets), validated on the last full season
xpfpl predict                      # top xP picks per position for the next 5 GWs
xpfpl recommend --team-id 1234567  # your team, transfers, captain and chip advice
xpfpl validate                     # score the model on a season it has never seen
xpfpl compare                      # train every model on the same season and rank them
xpfpl backtest --season 2024-25    # replay a whole season and count the points
xpfpl tune                         # set the config.py tuning parameters from replayed seasons (hours)
xpfpl scorecard                    # score this season's saved forecasts against the results so far
xpfpl export                       # write the website's data (web/public/data/)
xpfpl publish                      # export, build the website and push it to GitHub Pages
xpfpl archive                      # copy everything downloaded into archive/ (fetch/markets do this as they go)
```

Your team id is the number in the URL of your FPL *Points* page
(`fantasy.premierleague.com/entry/<team-id>/event/<gw>`).

Useful options for `recommend`:

| Option | Meaning |
| --- | --- |
| `--horizon 5` | how many gameweeks to plan over (see [Choosing a horizon](#choosing-a-horizon)) |
| `--max-hits 1` | allow up to this many -4 transfers |
| `--free-transfers N` | override the estimated free transfers |
| `--bank 1.5` | override money in the bank (£m) |
| `--model gbm` | use another model (see [Models](#models)) |
| `--no-plan` | just this week's transfers, instead of a route through the horizon |
| `--sensitivity 20` | re-solve 20 times with every player's xP randomly off by ~25%, and report how often each transfer and captain wins: is the advice robust or a coin-flip? |

Without `--team-id`, `recommend` builds the best squad from scratch within `--budget` (default £100m).

### Choosing a horizon

The horizon (`--horizon`, or **Horizon (GWs)** on the Plan Ahead tab) is how many gameweeks the
optimiser adds up when it judges a squad. Each week further out counts for less (`DISCOUNT = 0.8`):
this week counts fully, next week 0.8, the week after 0.64, and so on. A longer horizon makes it
favour players with good runs of fixtures over one good week.

**Leave it at 3 unless you have a reason to change it.** 3 is the default because it scored best
when whole seasons were replayed (see [Tuning](#tuning)), not because it was a guess:

- **Forecasts fade fast.** How well the model ranks players drops each week further out: 0.55
  one gameweek ahead, 0.45 two ahead, 0.37 three ahead. By week 4 or 5 the xP is mostly
  season-long averages, with little real information about that week.
- **A longer horizon adds noise.** It lets those weak late-week guesses pull transfers towards
  players who only look good far out. Replaying 2023-24 and 2024-25, a 3-week horizon beat 5
  and 8 by about a hundred points a season.
- It also solves about 20 times faster than 8.

When a different horizon makes sense:

| Situation | Horizon | Why |
| --- | --- | --- |
| Normal week | **3** | The tuned default |
| Free Hit week | **1** | The squad reverts afterwards, so only that week counts |
| Planning a Wildcard | **4-6** | You keep that squad for many weeks, so fixture runs matter more |
| Blank or double gameweek within the next ~5 weeks | **Long enough to include it** | Otherwise the plan can't see it. This logic hasn't been checked against a real blank or double gameweek yet |
| Banking free transfers for a later move | **Long enough to cover it** | So rolling the transfer shows its value |

A quick check: run the plan at 3 and again at 5. If this week's recommended transfer is the
same, it's robust and you can act on it. If it changes, the extra weeks are driving the move,
and those are the weeks the model predicts worst, so trust the 3-week answer unless you know
something it doesn't, like a coming double gameweek. Only this week's transfer is ever real
anyway: the later weeks get re-planned with fresh data every gameweek.

## Models

`--model` picks one anywhere it appears (`predict`, `recommend`, `train`, `backtest`, `tune`, the
dashboard). Each trains to its own file in `models/`, so they can coexist and be compared.

| Model | What it is |
| --- | --- |
| `mlp` | A 128-64 multi-layer perceptron on the rolling features. The default. |
| `components` | One head per scoring component - will he play, play 60 minutes, score, assist, keep a clean sheet, make saves, earn bonus - combined with FPL's own scoring rules (`xpfpl/scoring.py`). Trained with the loss that suits each head: cross-entropy for the yes/no questions, Poisson for counts. Its xP comes with a breakdown: "5.1, of which 1.8 from goals". |
| `embed` | The MLP plus a learned vector per player and per club (`nn.Embedding`), so it can hold an opinion beyond recent form. |
| `sequence` | A GRU reading the last six matches *in order*, so 2-2-12 and 5-4-3 look different even though both average 5. |
| `gbm` | LightGBM on the same features: the tabular benchmark the neural nets have to beat. `pip install -e ".[gbm]"`. |
| `xmins` | Expected minutes first: a softmax over *no minutes / a cameo / 60+*, then the points a cameo and a full game are worth. xP = P(cameo) x points + P(60+) x points. Also reports expected minutes and the chance of playing, shown in `predict` output and the dashboard. |
| `ensemble` | The average of `mlp`, `gbm` and `xmins` (without `gbm` if LightGBM isn't installed). |
| `baseline` | Each player's last-5-match average. No learning; the yardstick. |

```bash
xpfpl train --model components     # trains and validates, saves models/xp_components.pt
xpfpl compare                      # all of them on the same held-out season -> models/comparison.json
```

Set `MODEL` in `src/xpfpl/config.py` to change the default everywhere.

## Betting markets

`xpfpl markets` are prediction-market odds from Polymarket for
every Premier League match since 2024-25, and saves them to `data/processed/`. Each match comes
with several markets: the result (from 2024-25), total goals, each side's goals and both teams to
score (from 2025-26), and anytime goalscorer (from January 2026). There are also season-long
markets. `predict` and `recommend` fetch the live odds for the next gameweek themselves.

- **Priced at the FPL deadline.** A finished match's odds are rebuilt from its price history as
  they stood at that gameweek's deadline, never at kick-off: odds move on team news after the
  deadline, which a manager can't act on.
- **Expected goals from the odds.** Each side's expected goals are fitted to all of a match's goal
  markets at once (a small batched Poisson fit), which also gives clean-sheet chances.
- **In the model:** the market's expected goals, clean-sheet and win chances are features. Where
  there are no odds (before 2024-25, and any gameweek after the next one) they fall back to the
  model's own team ratings, with a flag saying which is which.

How good are the odds? At the deadline, the market predicts results better than the model's team
ratings, and goals and clean sheets about as well:

| Season | Goals RMSE (market / ours) | Clean-sheet Brier (market / ours) | Win log loss (market / ours) |
| --- | --- | --- | --- |
| 2024-25 (result markets only) | 1.192 / 1.183 | 0.176 / 0.172 | **0.593** / 0.595 |
| 2025-26 | **1.083** / 1.093 | **0.178** / 0.180 | **0.595** / 0.603 |

**Team news in the odds.** Injuries and rotation reach the market before FPL's flags. A player
whose anytime-goalscorer odds are 6% or less at the deadline has almost always been ruled out: on
2025-26, 3 of 62 such players played, against about 75% of everyone else, and the model (which
had given them a 23% chance of playing) over-predicted them by about half a point each. So
`predict` cuts that player's next-gameweek xP to 10% (`config.MARKET_OUT_XP_FACTOR`), and the
dashboard flags him as "Likely out".

Adding the team odds to the model is close to neutral on 2025-26: the component model gains the most (RMSE
2.638 -> 2.623, since it predicts clean sheets and goals directly), the others move by 0.002 or
less either way. They'll matter more as the market history grows (2024-25 only had result
markets to learn from). The anytime
goalscorer odds are shown in the dashboard but deliberately left out of the model: on 2025-26 they
priced scorers about 60% too high and ranked them barely better than chance (AUC 0.58, against
0.73 for the model's own goal predictions), and most of those markets hardly trade. Their one use
is the "ruled out" signal above.

## Tuning

The horizon, discount, bench weight, free-transfer value, hit allowance, price weight and chip
thresholds all change the season total, so they are set by measurement rather than by taste:

```bash
xpfpl tune --seasons 2023-24,2024-25     # a few hours; writes data/backtests/tuning.json
```

`tune` walks one small group of related tuning parameters at a time (coordinate descent), replays every
season in the list with each candidate value, keeps whatever scores most points, and moves on.
It prints the `config.py` block to paste in - deliberately not writing it for you, so the numbers
get a human glance first. The dashboard's Guide tab charts what each tuning parameter turned out to be worth.

The values in `config.py` come from a run over 2023-24 and 2024-25 with the MLP. What each stage
chose, and what it was worth (the gap between the best and worst value tried, in points a season):

| Stage | Chose | Was | Points a season | Worth |
| --- | --- | --- | --- | --- |
| horizon and discount | `HORIZON = 3`, `DISCOUNT = 0.8` | 5, 0.85 | 2162 | 108 |
| transfer planning | `PLAN_TRANSFERS = False`, `MAX_HITS = 0` | - | 2174 | 30 |
| free transfer value | `FT_VALUE = 3.0` | 1.5 | 2192 | 40 |
| bench weight | `BENCH_WEIGHT = 0.05` | 0.1 | 2234 | 43 |
| price changes | `PRICE_WEIGHT = 1.0` | 0.0 | 2258 | 68 |

Reading that as a whole: **every original guess was beaten**, and the tuning parameters are worth roughly as
much as the choice of model. Three results are worth calling out.

- A **3-gameweek horizon beats 5 and 8**. Predictions five weeks out are too vague to plan
  around, and the longer horizons also make the optimiser far slower.
- **Planning transfers week by week doesn't pay** - it is implemented and available
  (`plan_transfers=True`, the `--no-plan` flag inverts it), and the replay prefers the simpler
  mode by a small margin. Worse, the more hits the planner is allowed, the *worse* it does
  (2172 at zero hits, 2145 at two), because it schedules hits on predictions that don't survive
  contact with the next gameweek. With planning off, the optimiser never takes a hit at all.
- **Valuing price rises is worth ~68 points a season**, the second biggest effect found - but
  only at a small weight. `PRICE_WEIGHT = 1.0` treats £1m of expected value as one point, which
  breaks ties towards a player about to rise; at 3.0 it starts distorting the squad and loses
  38 of those points again.

The four chip thresholds are still the original guesses: those stages of the search haven't been
run to completion. `xpfpl tune --stages wildcard` does one of them.

## How good is it?

Two different questions, two different tools. Both write their results where the dashboard's
**Guide** tab can chart them.

### Accuracy on a season it has never seen

`xpfpl train` (and `xpfpl validate`, which does the same without overwriting the saved model)
trains on 2016-17 to 2024-25, predicts every match of 2025-26 and writes
`models/validation.json`. `xpfpl compare` does it for every model at once
(`models/comparison.json`). Points off per player per match, for players getting minutes,
lower is better:

| Model | RMSE | MAE | R² | Rank corr | Captain test | Train |
| --- | --- | --- | --- | --- | --- | --- |
| sequence (GRU) | **2.600** | 1.731 | **0.183** | **0.561** | 6.4 | 54s |
| ensemble (mlp+gbm+xmins) | 2.601 | 1.731 | 0.182 | 0.558 | 5.7 | 81s |
| xmins | 2.608 | 1.748 | 0.178 | 0.555 | 6.0 | 44s |
| mlp | 2.612 | 1.746 | 0.175 | 0.546 | 6.5 | 36s |
| gbm | 2.615 | 1.721 | 0.174 | 0.553 | 5.0 | 3s |
| embed | 2.637 | 1.780 | 0.160 | 0.523 | 6.3 | 23s |
| components | 2.638 | **1.718** | 0.159 | 0.543 | 4.5 | 51s |
| baseline (5-match average) | 2.882 | 1.965 | -0.004 | 0.408 | 4.1 | - |
| FPL's own xP | 3.433 | 2.119 | -0.424 | 0.350 | 2.4 | - |
| *perfect-model ceiling* | *2.42 (2.37-2.47)* | *1.60* | *0.21* | | | |

*FPL's own xP* is the expected-points figure the official game shows for each player, taken as
it stood before each match. It is essentially recent form, so it's a yardstick rather than a rival.

The *perfect-model ceiling* takes the component model's probabilities as the truth, simulates every
match 200 times, and scores the true expectation against each simulated week. Even that scores
RMSE ~2.4 and R² ~0.2, because most of a week's points are luck. The gap between it and the real
models is the room left to improve.

- **MAE** is the typical miss; **RMSE** punishes the big ones, so the gap between them is a
  measure of hauls and blanks.
- Judge a model on **players getting minutes** (played at least once in their previous five
  matches), because that's the pool you pick from. On *all* rows every model looks better (the
  MLP scores 1.953 RMSE / 0.994 MAE against the baseline's 2.111 / 1.052), but over half of
  those rows are players who never came on and scored 0 or 1: easy marks.
- The **captain test** is the football-shaped version: captain whoever the model rates highest
  in the league each week and count what they scored. Perfect hindsight would be 17.2 a week and
  a typical starting player 2.2, so everything here beats guessing and nothing here is clairvoyant.
- **R²** is the share of the week-to-week variation in points the predictions explain, and **rank
  corr** (Spearman, within each gameweek) is whether the players are in the right order - which is
  all the optimiser needs.
- The trained models sit within a few hundredths of a point of each other - a much smaller gap
  than the one between all of them and the baseline. The ceiling is in the features, not the
  architecture: the September 2026 feature work moved *every* model by ~0.05 RMSE, more than any
  change of model ever did, and almost all of it came from one signal - the crowd's net transfers
  before the deadline (plus ownership), which carries the team news the stats can't.
- Forecasts fade quickly: RMSE 2.61, 2.70 and 2.77 for forecasts made 1, 2 and 3 gameweeks ahead
  (`xpfpl validate --horizons 3`), which is why a 3-gameweek horizon with a 0.8 discount works.
- The captain test moves by more than a point a gameweek between models that are otherwise
  indistinguishable, because it is one pick over 38 weeks. Don't read much into it.
- The component model is the one worth keeping for a different reason: its xP comes apart into
  readable pieces (Haaland's 6.7 = 1.9 minutes + 3.3 goals + 0.6 assists + 1.0 bonus - 0.1 cards),
  which `xpfpl predict --model components` prints.
- Accuracy is not the last word: the season replays below rank the models differently, and by
  bigger margins. `config.MODEL` stays on `mlp` (LightGBM is an optional dependency),
  but see the note there before taking that as a recommendation.
- The report also holds a calibration curve (does 6 xP really mean six points?), accuracy per
  position and per gameweek, and the error distribution.

### Points on the board: replaying a season

`xpfpl backtest` replays a past season from the first deadline to the last. Each week it rebuilds
form from matches already played, predicts the next `--horizon` gameweeks from that snapshot,
runs the same optimiser with the free transfers and bank it has accrued, then scores the week for
real, with auto-subs, captaincy and hits.

```bash
xpfpl backtest --season 2024-25                        # ~2200-2300 points over 38 GWs
xpfpl backtest --season 2024-25 --chips                # also play chips, at the config.py thresholds
xpfpl backtest --sweep horizon=3,5,8 discount=0.8,0.9  # one run per combination, ranked by points
```

Replaying two seasons (horizon 3, discount 0.8, planning transfers, no chips) puts the models in
a different order from the accuracy table, and by much bigger margins:

| Model | 2023-24 | 2024-25 | Average |
| --- | --- | --- | --- |
| `gbm` | 2461 | 2300 | **2381** |
| `ensemble` | 2287 | 2447 | 2367 |
| `mlp` | 2176 | 2261 | 2219 |
| `xmins` | 2170 | 2154 | 2162 |

LightGBM wins again (it also won before the feature work, 2305 to the MLP's 2210, at the older
settings), and the ensemble - two-thirds PyTorch - is within noise of it. Two lessons, both worth
more than the numbers: a 0.005 difference in RMSE is noise, but which players end up in the squad
is not; and the way to choose between models here is to play the seasons out, not to rank error
metrics. If you want the points, set `MODEL = "ensemble"` (keeps the PyTorch models in the loop)
or `"gbm"` in `src/xpfpl/config.py`, with `pip install -e ".[gbm]"`.

### The live record

Before each deadline, `predict` and `recommend` save their forecasts to
`data/predictions/<season>/`. After the gameweek, `xpfpl fetch` then `xpfpl scorecard` scores
them against what happened - MAE, RMSE, R², rank correlation and bias per gameweek, plus bias by
position, the early warning for a rule change the history can't teach (like 2026-27's new bonus
points system). Forecasts made in advance can't be quietly revised, so this is the honest
number to watch.

Results land in `data/backtests/` as CSV and JSON. This is what the tunable parameters should be set
from, because it measures them in points rather than in decimals of RMSE.

Known simplifications: no injury flags (they aren't in the historical data, so the replay fields
players a real manager would have avoided), the squad's value doesn't benefit from price rises,
and chip windows follow the current two-halves rule. Treat a total as a floor and the comparison
between settings as the useful part. A replay is also chaotic - one different captain pick in
October changes everything after it - so differences of 20 or 30 points a season are noise, and
only the consistent ones (across both seasons) are worth acting on.

## Each gameweek

1. **After the gameweek is complete** (the sidebar says so, or FPL shows the data as confirmed),
   run the sidebar's steps 1-3, or:
   ```bash
   xpfpl fetch       # new results; also copies in the Action's deadline snapshots (see below)
   xpfpl markets     # odds for the played matches
   xpfpl train
   ```
   All three add their data to `archive/` as they go.
2. **Before the deadline**: `xpfpl recommend --team-id <id>` (or the Plan Ahead tab). This saves
   the forecast that the website's xP and the [live record](#the-live-record) are scored on.
3. **Update the website**: the sidebar's step 4, or `xpfpl publish`. It's live a minute or two later.
4. **Commit the archive** every week or so. `main` only accepts pull requests, so the new files
   in `archive/` stay on your machine until you merge them (on Windows PowerShell, run the
   commands one per line; `&&` doesn't work there):
   ```bash
   git checkout main
   git pull
   git checkout -b archive-gw06
   git add archive
   git commit -m "Archive to GW6"
   git push -u origin archive-gw06
   ```
   Open the pull request from the link `git push` prints, merge it, then
   `git checkout main`, `git pull` and `git branch -d archive-gw06`.

Nothing is needed for the deadline snapshots: the scheduled Action takes them.

**When a season ends**, add it to `config.HISTORY_SEASONS` (it loads from the archive from then
on), and commit the archive as above before FPL resets for the new season, because the API
drops the old season's match-by-match history.

### Branches

| Branch | What it is | You |
| --- | --- | --- |
| `main` | The code and the archive. Protected: changes only through pull requests | Work on a branch, merge by pull request |
| `gh-pages` | The built website that GitHub Pages serves. `xpfpl publish` replaces it with one fresh commit each time | Never edit or merge it |
| `deadline-snapshots` | Where the scheduled Action pushes each pre-deadline snapshot | Never merge it: `xpfpl fetch` copies the snapshots into `archive/` |

## Website

A public, read-only look back at the season, served by GitHub Pages from the `gh-pages` branch
(https://amparaj.github.io/xpfpl/). Its pages, in order: **About** (where it opens: the project and
how the model forecasts, picks a team and is tested, in plain language, with the latest accuracy
figures), **Model Accuracy**, **Gameweeks** (every result, each player's points against the xP
forecast), **Players**, **Markets** (the betting odds at each deadline against what happened),
**My Team** (one FPL team's season: picks against the hindsight-best XI, transfers, chips) and
**Data** (the archive). It doesn't train or plan.
The Markets page's upcoming-match odds, price histories and season markets come from
`odds.json` on the `odds` branch, which a scheduled GitHub Action (`.github/workflows/odds-snapshot.yml`)
fetches from Polymarket on a schedule (Actions -> Odds snapshot -> Run workflow refreshes it by
hand). The page reads it through raw.githubusercontent.com. The
browser never calls Polymarket itself, because it's blocked on some networks (Australian ISPs, for
one), and the file isn't on `gh-pages` because every push there rebuilds the site (Pages allows
about 10 builds an hour). Played matches' price charts come from the export (`market_history/`,
built from the archive). To see live odds locally, run
`node web/scripts/odds-snapshot.ts web/public/data/meta.json web/public/data/odds.json`
(Node 23.6+) after `xpfpl export`.

```bash
xpfpl publish          # export -> build web/ -> push web/dist to gh-pages
xpfpl publish --build-only && npm --prefix web run preview   # look at it locally first
```

`xpfpl export` writes `web/public/data/` (about 13 MB of JSON, most of it played matches' price histories; not committed). `xpfpl publish`
builds the React app in `web/` (Vite + TypeScript, charts with Observable Plot; needs Node.js)
and force-pushes `web/dist/` to `gh-pages` as a single commit, so the site's data never piles up
in the repo's history. The team shown is the one saved in the dashboard (or `--team-id`). To turn
the site on the first time: GitHub -> Settings -> Pages -> Deploy from a branch -> `gh-pages`, `/ (root)`.

For development: `npm --prefix web install`, then `npm --prefix web run dev`.

## Archive

[`archive/`](archive/README.md) is a zstd-Parquet copy of every source the pipeline reads, kept in
git (about 15 MB, plus about 5-10 MB a season): vaastav's seasons 2016-17 to 2025-26, this season
from the FPL API one file per gameweek, every player's price, news and injury flag shortly before
each deadline, the saved forecasts, and Polymarket's events and price histories for every played
match since 2024-25. `fetch` reads the archive first and only downloads what it doesn't have.
When a season ends, add it to `config.HISTORY_SEASONS` and it loads from the archive like
vaastav's seasons, so vaastav is no longer needed. `fetch`, `markets` and `predict` add to it as
they go. A GitHub Action (`.github/workflows/deadline-snapshot.yml`) takes the pre-deadline
snapshot every gameweek. `main` only accepts pull requests, so the Action pushes the snapshots to
the `deadline-snapshots` branch; `xpfpl fetch` (and `xpfpl archive`) copies them into `archive/`,
and they reach `main` with your next pull request. How and when to commit the archive is in
[Each gameweek](#each-gameweek).

## Project layout

```
src/xpfpl/
  config.py          paths, rules and tunable parameters (horizon, discount, chip thresholds)
  data/api.py        FPL API client
  data/history.py    archive (vaastav) + API -> matches.parquet
  data/archive.py    the git-tracked archive of every source (archive/)
  features.py        feature engineering (training rows and upcoming fixtures)
  teams.py           team attack/defence ratings (a Poisson model refitted every gameweek)
  data/markets.py    betting-market odds (Polymarket), priced at FPL deadlines
  market_view.py     the dashboard's Markets tab
  scoring.py         FPL's scoring rules, applied to real or predicted stats
  models/__init__.py the model registry: fit / load / save, one interface
  models/trainer.py  the training loop the PyTorch models share (batches, early stopping)
  models/baseline.py 5-match average baseline
  models/mlp.py      the default MLP
  models/components.py  a head per scoring component, combined with FPL's rules
  models/embed.py    MLP + player and club embeddings
  models/sequence.py GRU over the last six matches
  models/gbm.py      LightGBM benchmark
  models/minutes.py  xmins: expected minutes, then points given the minutes
  models/ensemble.py the average of mlp, gbm and xmins
  predict.py         xP per player per upcoming gameweek
  prices.py          expected price changes (form table + live transfer momentum)
  optimise.py        squad / transfer / lineup optimiser (PuLP), week by week
  myteam.py          your squad, selling prices, bank, free transfers, chips used
  chips.py           chip rules
  validate.py        held-out-season accuracy report (models/validation.json)
  scorecard.py       the live record: saved forecasts scored against results
  backtest.py        replay a past season deadline by deadline; sweep the tuning parameters
  tune.py            search the config.py tuning parameters with the backtest
  export.py          the website's data (web/public/data/) and publishing to gh-pages
  cli.py             command line entry point
web/                 the public website (React + TypeScript, Vite)
archive/             every source, compressed (see archive/README.md)
tests/               scoring rules, features, every model, optimiser rules, prices, backtest,
                     tuning, validation, the live scorecard, market odds, archive, export
```

## Caveats

- Free transfers are estimated by replaying your season's transfers. Check them against the FPL
  site and use `--free-transfers` if they differ.
- The plan for later gameweeks is a route, not a commitment: it assumes today's predictions and no
  new information, and is re-planned from scratch every week. Only this week's move is real.
- Price changes are predicted from form and (live) transfer momentum, not from FPL's real hidden
  thresholds. They are a tie-breaker between similar players, nothing more.
- Scoring rules change between seasons (for example, defensive contribution points from 2025-26).
  The model sees `xg_era`/`dc_era` flags, but older seasons are only a rough guide to today's scoring.
- The model's output is advice. Check the news, team sheets and your own judgement before the deadline.

## Data sources

- [FPL API](https://fantasy.premierleague.com/api/bootstrap-static/): live and current-season data
- [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League): historical gameweek data
- [Polymarket](https://polymarket.com) (Gamma and CLOB APIs): betting odds
- [fantasynutmeg.com/history](https://www.fantasynutmeg.com/history): handy for sanity-checking past seasons
