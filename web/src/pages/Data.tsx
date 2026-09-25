import { useSite } from "../site";

const FILES = [
  ["fpl/<season>/gws.parquet", "One row per player per match, with points and every stat, 2016-17 to 2025-26 (vaastav's merged_gw.csv)."],
  ["fpl/<season>/gws/gwNN.parquet", "The same from 2026-27, one file per gameweek, from the FPL API."],
  ["fpl/<season>/players.parquet", "Who's who: the season's player id, the persistent code, names, position and club."],
  ["fpl/<season>/teams.parquet, fixtures…", "Clubs, fixtures and scores, and gameweek deadlines and averages."],
  ["fpl/<season>/deadlines/gwNN.parquet", "Every player shortly before each deadline: price, injury news, chance of playing, ownership, transfers."],
  ["predictions/<season>/gwNN_<model>.parquet", "This model's forecasts, saved before each deadline."],
  ["polymarket/<season>/events, prices", "Every Polymarket market on each played match, and its price history up to the FPL deadline (from 2024-25)."],
  ["polymarket/outrights/<date>.parquet", "Season-long markets: title, top four, relegation, top scorer…"],
];

export default function DataPage() {
  const site = useSite();
  const repo = site.meta.repo;
  return (
    <>
      <h2>The data</h2>
      <p className="lede">
        Everything on this site comes from an archive of Fantasy Premier League data back to 2016-17 and betting odds
        from 2024-25, kept in the project's GitHub repository as compressed Parquet files. It's free to use.
      </p>
      {repo && (
        <p>
          <a href={`${repo}/tree/main/archive`}>Browse the archive on GitHub</a> · <a href={repo}>The code</a>
        </p>
      )}
      <div className="table-wrap">
        <table>
          <thead><tr><th>File</th><th>What's in it</th></tr></thead>
          <tbody>
            {FILES.map(([f, d]) => (
              <tr key={f}><td><code>{f}</code></td><td style={{ whiteSpace: "normal" }}>{d}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <h3>Reading it</h3>
      <pre className="card" style={{ overflowX: "auto", fontSize: 13 }}>{`import pandas as pd
gws = pd.read_parquet("archive/fpl/2025-26/gws.parquet")

-- or DuckDB, straight from GitHub:
select * from '${repo ? repo.replace("github.com", "raw.githubusercontent.com") + "/main" : "…"}/archive/fpl/2025-26/gws.parquet' limit 10;`}</pre>
      <h3>Sources</h3>
      <ul>
        <li>Past seasons: <a href="https://github.com/vaastav/Fantasy-Premier-League">vaastav/Fantasy-Premier-League</a>. Please credit it.</li>
        <li>This season: the public <a href="https://fantasy.premierleague.com">Fantasy Premier League</a> API. The data belongs to the Premier League.</li>
        <li>Odds: <a href="https://polymarket.com">Polymarket</a>. Upcoming matches on the Markets page are fetched from its public API every 30 minutes.</li>
      </ul>
    </>
  );
}
