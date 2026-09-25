// Cup and European matches (data/cups.py): the midweek fixtures that sit between gameweeks.
// The export files each one under the gameweek it comes before.

import type { MidweekMatch } from "./data";
import type { Site } from "./site";

export const COMPETITIONS: Record<string, { short: string; name: string }> = {
  "champions-league": { short: "UCL", name: "Champions League" },
  "europa-league": { short: "UEL", name: "Europa League" },
  "conference-league": { short: "UECL", name: "Conference League" },
  "efl-cup": { short: "EFL", name: "EFL Cup" },
  "fa-cup": { short: "FA", name: "FA Cup" },
  "uefa-super-cup": { short: "USC", name: "UEFA Super Cup" },
  "community-shield": { short: "CS", name: "Community Shield" },
};

export const competition = (t: string) => COMPETITIONS[t] ?? { short: t, name: t };

/** A club's midweek matches before gameweek `gw` (by FPL team id). */
export function clubMidweek(site: Site, team: number, gw: number): MidweekMatch[] {
  const code = site.team.get(team)?.code;
  return site.midweek.filter((m) => m.gw === gw && code !== undefined && (m.home_code === code || m.away_code === code));
}

/** The name to show for one side: our short club name for a Premier League club, else the source's. */
export function side(site: Site, m: MidweekMatch, home: boolean): string {
  const code = home ? m.home_code : m.away_code;
  const team = code === null ? undefined : site.teamByCode.get(code);
  return team?.name ?? (home ? m.home_name : m.away_name);
}

/** "Arsenal v Lille" or, once played, "Arsenal 2–1 Lille". */
export function describe(site: Site, m: MidweekMatch): string {
  const score = m.finished && m.home_score !== null ? ` ${m.home_score}–${m.away_score} ` : " v ";
  return `${side(site, m, true)}${score}${side(site, m, false)}`;
}

/** Next week's rotation groups (the forecast's `rotation`), in words. */
export const ROTATION: Record<string, string> = {
  regular_rested: "Regular starter, rested midweek",
  regular_part: "Regular starter, played part of the midweek match",
  regular_full: "Regular starter, played the whole midweek match",
  squad_unused: "Squad player, not used midweek",
  squad_played: "Squad player, played midweek",
};

/** The weekday of a kick-off in UK time ("Tue"): a European evening is already the next morning
 * in Australia, and "UCL Wed" for a Tuesday-night match would read wrong. */
export const matchDay = (iso: string | null) =>
  iso ? new Date(iso).toLocaleDateString("en-GB", { weekday: "short", timeZone: "Europe/London" }) : "";
