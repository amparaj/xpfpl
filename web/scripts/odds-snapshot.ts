// Saves Polymarket's live odds as <site>/data/odds.json, for the Markets page to read from the
// site's own domain. Run every 30 minutes by .github/workflows/odds-snapshot.yml, on GitHub's
// runners, because Polymarket is blocked on some networks (Australian ISPs among them).
//
//   node web/scripts/odds-snapshot.ts <site folder>     (e.g. web/public/data/.. for `npm run dev`)
//
// Needs Node 23.6+ (runs the TypeScript directly) and the site's data/meta.json for the club codes.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fetchHistory, liveOdds, liveOutrights, type MatchHistory, type OddsSnapshot } from "../src/polymarket.ts";

const site = process.argv[2] ?? "web/public";
const meta = JSON.parse(readFileSync(join(site, "data", "meta.json"), "utf-8"));
const now = new Date();
const [{ matches, scorers }, outrights] = await Promise.all([liveOdds(meta.polymarket_teams), liveOutrights()]);
const history: Record<string, MatchHistory> = {};
for (const m of matches) history[m.slug] = await fetchHistory(m, now);   // one match at a time: gentle on the API
const snapshot: OddsSnapshot = { fetched_at: now.toISOString(), matches, scorers, outrights, history };
writeFileSync(join(site, "data", "odds.json"), JSON.stringify(snapshot));
console.log(`${matches.length} matches, ${scorers.length} scorer markets, ${outrights.length} season outcomes.`);
