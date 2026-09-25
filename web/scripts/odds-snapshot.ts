// Saves Polymarket's live odds as a JSON file for the Markets page. Run by
// .github/workflows/odds-snapshot.yml on GitHub's runners, because Polymarket is blocked on some
// networks (Australian ISPs among them); the page reads the file from GitHub instead.
//
//   node web/scripts/odds-snapshot.ts <meta.json> <odds.json> [--when-due <previous odds.json>]
//   node web/scripts/odds-snapshot.ts web/public/data/meta.json web/public/data/odds.json   (for `npm run dev`)
//
// With --when-due it only fetches when the odds are worth refreshing (see `due`), and otherwise
// exits without writing anything. Needs Node 23.6+ (runs the TypeScript directly) and the site's
// meta.json for the club codes, deadlines and kick-offs.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { fetchHistory, liveOdds, liveOutrights, type MatchHistory, type OddsSnapshot } from "../src/polymarket.ts";

const BUSY_BEFORE_DEADLINE = 48;   // hours: refresh on every run from this long before a deadline...
const BUSY_AFTER_LAST_KICKOFF = 2; // ...until this long after the gameweek's last kick-off
const QUIET_EVERY = 3;             // hours between refreshes the rest of the time
const HOUR = 3600e3;

interface Meta {
  polymarket_teams: Record<string, number>;
  events: { id: number; deadline: string }[];
  fixtures: { gw: (number | null)[]; kickoff: (string | null)[] };
}

/** Why the odds are worth refreshing now, or null if they aren't. */
function due(meta: Meta, previous: string | undefined, now: number): string | null {
  for (const ev of meta.events) {
    const kickoffs = meta.fixtures.kickoff.filter((k, i) => k && meta.fixtures.gw[i] === ev.id).map((k) => Date.parse(k!));
    const deadline = Date.parse(ev.deadline);
    const last = kickoffs.length ? Math.max(...kickoffs) : deadline;
    if (now >= deadline - BUSY_BEFORE_DEADLINE * HOUR && now <= last + BUSY_AFTER_LAST_KICKOFF * HOUR) return `GW${ev.id} is on`;
  }
  if (!previous || !existsSync(previous)) return "no odds saved yet";
  const age = (now - Date.parse(JSON.parse(readFileSync(previous, "utf-8")).fetched_at)) / HOUR;
  // A little early is fine: scheduled runs drift, and waiting for the next one would add 5 minutes.
  return age >= QUIET_EVERY - 0.1 ? `the odds are ${age.toFixed(1)} hours old` : null;
}

const args = process.argv.slice(2);
const flag = args.indexOf("--when-due");
const previous = flag >= 0 ? args.splice(flag, 2)[1] : undefined;
const [metaPath = "web/public/data/meta.json", outPath = "web/public/data/odds.json"] = args;
const meta: Meta = JSON.parse(readFileSync(metaPath, "utf-8"));
const now = new Date();
if (flag >= 0) {
  const reason = due(meta, previous, now.getTime());
  if (!reason) {
    console.log(`Not due: no deadline within ${BUSY_BEFORE_DEADLINE} hours and the odds are under ${QUIET_EVERY} hours old.`);
    process.exit(0);
  }
  console.log(`Refreshing: ${reason}.`);
}
const [{ matches, scorers }, outrights] = await Promise.all([liveOdds(meta.polymarket_teams), liveOutrights()]);
const history: Record<string, MatchHistory> = {};
for (const m of matches) history[m.slug] = await fetchHistory(m, now);   // one match at a time: gentle on the API
const snapshot: OddsSnapshot = { fetched_at: now.toISOString(), matches, scorers, outrights, history };
writeFileSync(outPath, JSON.stringify(snapshot));
console.log(`${matches.length} matches, ${scorers.length} scorer markets, ${outrights.length} season outcomes.`);
