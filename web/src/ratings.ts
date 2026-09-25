// The model's own odds for upcoming matches, and matching betting markets' player names to FPL's.

import type { Player } from "./data";
import { poissonWin, type LiveScorer } from "./polymarket";
import type { Site } from "./site";

/** Our Odds for an upcoming match (clubs by persistent code): the club ratings exported for the
 * next gameweek. Expected goals each side and each side's chance of winning. */
export function ours(site: Site, home: number, away: number) {
  const r = site.meta.ratings_next;
  if (!r || !("clubs" in r)) return { ours_home: null, ours_away: null, ours_home_win: null, ours_away_win: null };
  const [ha, hd] = r.clubs[String(home)] ?? r.prior;
  const [aa, ad] = r.clubs[String(away)] ?? r.prior;
  const gf = Math.exp(r.mu + r.home + ha - ad), ga = Math.exp(r.mu + aa - hd);
  return { ours_home: gf, ours_away: ga, ours_home_win: poissonWin(gf, ga), ours_away_win: poissonWin(ga, gf) };
}

const strip = (s: string) => s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z ]/g, "").trim();

/** The FPL player a scorer market names, among the two clubs' players (as markets.match_players does). */
export function matchPlayer(site: Site, s: LiveScorer): Player | undefined {
  const clubs = new Set([site.teamByCode.get(s.home_code)?.id, site.teamByCode.get(s.away_code)?.id]);
  const key = strip(s.player);
  const hits = site.players.filter((p) => {
    if (!clubs.has(p.team)) return false;
    const full = strip(`${p.first_name} ${p.second_name}`);
    const parts = full.split(" ");
    return [strip(p.web_name), full, `${parts[0]} ${parts[parts.length - 1]}`, strip(p.second_name)].includes(key);
  });
  return hits.length === 1 ? hits[0] : undefined;
}
