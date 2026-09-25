# Archive

A compressed copy of every data source xP-FPL uses, kept in git so the history survives if a
source goes away. It holds Fantasy Premier League data from 2016-17 onwards and Polymarket odds
from 2024-25 onwards. All files are [Parquet](https://parquet.apache.org/) with zstd compression.
Read them with pandas (`pd.read_parquet`), DuckDB (`select * from 'archive/fpl/*/gws.parquet'`)
or any Parquet reader.

| Path | What | Source |
| --- | --- | --- |
| `fpl/<season>/gws.parquet` | One row per player per match: points and every stat. 2016-17 to 2025-26. | [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) `gws/merged_gw.csv` |
| `fpl/<season>/gws/gwNN.parquet` | The same, one file per gameweek, from 2026-27 on | FPL API `element-summary/{id}/` |
| `fpl/<season>/players.parquet` | Players: `id` (renumbered each season), persistent `code`, names, position (`element_type`), club | vaastav `players_raw.csv` / FPL API `bootstrap-static` |
| `fpl/<season>/teams.parquet` | Clubs: `id` (renumbered each season), persistent `code`, names | vaastav `teams.csv` / FPL API |
| `fpl/<season>/fixtures.parquet`, `fixtures/gwNN.parquet` | Fixtures, scores and per-match stats | vaastav `fixtures.csv` / FPL API `fixtures/` |
| `fpl/<season>/events.parquet` | Gameweeks: deadlines, average and highest scores, chip plays | FPL API |
| `fpl/<season>/deadlines/gwNN.parquet` | Every player shortly before gameweek NN's deadline: price, injury news, `chance_of_playing_next_round`, ownership, that week's transfers | FPL API `bootstrap-static` |
| `predictions/<season>/gwNN_<model>.parquet` | xP-FPL's forecasts, saved before each deadline | this project |
| `modelteam/<season>/gwNN.json` | The Model's Team: its transfers, XI, bench order, captain and chip for gameweek NN, saved before the deadline (`source` "live"), or filled in by the backtest for weeks before the live record began ("replay"). `next` is the squad, bank and free transfers it hands to the following week. | this project |
| `cups/<season>/fixtures/gwNN.parquet` | Cup and European matches (Champions/Europa/Conference League, EFL Cup) of each EPL club, filed under the FPL gameweek they come before, from 2025-26 on. `team_code` is FPL's club code; later gameweeks are fixtures not yet played | [olbauday/FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights) `By Gameweek/GWn/fixtures.csv` |
| `cups/<season>/minutes/gwNN.parquet` | Minutes per player (FPL `code`) in those matches, and whether he started | FPL-Core-Insights `playermatchstats.csv` |
| `polymarket/<season>/events/gwNN.parquet` | Every Polymarket event for each played match (result, goal totals, both teams to score, anytime scorer), as trimmed JSON | [Polymarket Gamma API](https://gamma-api.polymarket.com) |
| `polymarket/<season>/prices/gwNN.parquet` | Price histories for those markets up to the FPL deadline: `window` is `history_14d` (hourly, 14 days) or `history_3h_5m` (every 5 minutes, 3 hours). `t` is a Unix time; `p` is the probability of "Yes". A row with `t = -1` means no trades. | Polymarket CLOB API |
| `polymarket/outrights/<date>.parquet` | Season-long markets (title, top four, relegation, top scorer...) as priced that day | Polymarket |

Past seasons never change. The current season gains a few files every week, written by
`xpfpl fetch`, `xpfpl markets`, `xpfpl predict` and `xpfpl recommend`. The deadline snapshots are taken by a
scheduled GitHub Action (`.github/workflows/deadline-snapshot.yml`, which pushes them to the
`deadline-snapshots` branch until they're merged here), or by hand with
`xpfpl archive --deadline`.

Data before 2026-27 comes from vaastav's repository; please credit it if you use it. Cup and European
matches come from olbauday's FPL-Core-Insights. FPL data
belongs to the Premier League, and the odds belong to Polymarket.
