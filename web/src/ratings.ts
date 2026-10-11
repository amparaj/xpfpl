// The model's own odds for upcoming matches.

import { poissonWin } from "./polymarket";
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
