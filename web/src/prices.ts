// The live price tracker: prices.json from the `prices` branch, refreshed hourly by a scheduled
// GitHub Action (src/xpfpl/data/pricewatch.py). Progress is FPL's own figure: +100 rises, -100 falls.

export interface PricePlayer {
  id: number; code: number; name: string; team: number; team_code: number; pos: number;
  /** Price in tenths of £m, and its change this gameweek and this season (tenths). */
  cost: number; change_event: number; change_start: number;
  selected: number; net_event: number;
  /** FPL's progress to the next price change (-100 to +100), how fast it's moving per hour,
   * the progress projected at the next three price updates and FPL's likelihood of a move at each (-5 to +5). */
  pct: number | null; rate: number | null; proj: (number | null)[]; lik: (number | null)[];
  locked_until: string | null; status: string;
  /** Progress at each of `times` (the last three days of hourly runs). */
  h: (number | null)[];
}
export interface PriceChange {
  /** When the hourly run first saw it (null: before the log started). */
  t: string | null; gw: number | null; id: number; code: number; name: string; team_code: number; pos: number;
  from: number; to: number;
}
export interface PriceLog {
  updated: string; season: string; gw_current: number | null; gw_next: number | null; next_deadline: string | null;
  times: string[]; players: PricePlayer[]; changes: PriceChange[];
}

/** Where the live file is: the `prices` branch through raw.githubusercontent.com on github.io (it allows
 * requests from any site and caches for 5 minutes), else the copy `xpfpl export` wrote. */
export function pricesUrl(): string {
  const { hostname, pathname } = window.location;
  if (!hostname.endsWith(".github.io")) return "prices.json";
  const owner = hostname.slice(0, -".github.io".length), repo = pathname.split("/")[1];
  return `https://raw.githubusercontent.com/${owner}/${repo}/prices/prices.json`;
}

/** FPL's -5..+5 likelihood in words; the sign is the direction. */
export function likelihood(v: number | null | undefined): string {
  if (v === null || v === undefined) return "";
  const words = ["no move", "unlikely", "possible", "likely", "very likely", "certain"];
  const word = words[Math.min(Math.abs(Math.round(v)), 5)];
  return v === 0 ? word : `${v > 0 ? "rise" : "fall"} ${word}`;
}

/** A change's day in UK time ("Tue 14 Oct"): FPL updates prices overnight UK time. */
export function ukDay(iso: string): string {
  return new Date(iso).toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", timeZone: "Europe/London" });
}
