// Saves Polymarket's live odds as a JSON file for the Markets page. Run every 5 minutes by
// .github/workflows/odds-snapshot.yml, on GitHub's runners, because Polymarket is blocked on some
// networks (Australian ISPs among them); the page reads the file from GitHub instead.
//
//   node web/scripts/odds-snapshot.ts <meta.json> <odds.json>
//   node web/scripts/odds-snapshot.ts web/public/data/meta.json web/public/data/odds.json   (for `npm run dev`)
//
// Needs Node 23.6+ (runs the TypeScript directly) and the site's meta.json for the club codes.
import { readFileSync, writeFileSync } from "node:fs";
import { fetchHistory, liveOdds, liveOutrights, type MatchHistory, type OddsSnapshot } from "../src/polymarket.ts";

const [metaPath = "web/public/data/meta.json", outPath = "web/public/data/odds.json"] = process.argv.slice(2);
const meta = JSON.parse(readFileSync(metaPath, "utf-8"));
const now = new Date();
const [{ matches, scorers }, outrights] = await Promise.all([liveOdds(meta.polymarket_teams), liveOutrights()]);
const history: Record<string, MatchHistory> = {};
for (const m of matches) history[m.slug] = await fetchHistory(m, now);   // one match at a time: gentle on the API
const snapshot: OddsSnapshot = { fetched_at: now.toISOString(), matches, scorers, outrights, history };
writeFileSync(outPath, JSON.stringify(snapshot));
console.log(`${matches.length} matches, ${scorers.length} scorer markets, ${outrights.length} season outcomes.`);
