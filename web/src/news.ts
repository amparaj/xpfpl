// Team news: news.json from the `news` branch, refreshed hourly by a scheduled GitHub Action
// (src/xpfpl/data/news.py): FPL's news for every player, the managers' press conferences, the latest
// NewsNow headlines and a log of every change to FPL's news this season.

export interface NewsRisk { gw: number | null; what: string | null; notes: string | null; url: string | null }
export interface NewsPlayer {
  id: number; code: number; name: string; first: string | null; second: string | null;
  team: number; team_code: number; pos: number;
  /** FPL's status code (a, d, i, s, u, n) and chance of playing next round / this round (null = no flag). */
  status: string; chance: number | null; chance_this: number | null;
  /** FPL's news, its cause ("Hamstring injury"), what sort it is, and the return date it gives (ISO date). */
  news: string; reason: string; kind: "injury" | "doubt" | "suspension" | "left" | "other"; back: string | null;
  /** When FPL last changed the news, and the article it came from. */
  added: string | null; source: string | null;
  /** Known absences ahead, e.g. a loanee who can't face his parent club. */
  risks: NewsRisk[];
  selected: number; cost: number;
  /** What a press conference for the next gameweek said about him (null: nothing, or not yet for this week). */
  press: "OUT" | "DOUBT" | "IN" | null;
}
export interface PressClub {
  club: string; team: number | null; team_code: number | null;
  players: { name: string; status: "OUT" | "DOUBT" | "IN"; id: number | null; code: number | null }[];
  quotes: { text: string; by: string | null }[];
}
export interface Press { updated: string | null; fetched?: string; fresh: boolean; clubs: PressClub[] }
export interface NewsChange {
  /** FPL's time for the change (news_added) and when the hourly run saw it. */
  t: string | null; seen: string; gw: number | null; id: number; code: number; name: string; team_code: number; pos: number;
  status: string; chance: number | null; news: string; back: string | null; source: string | null;
}
/** A NewsNow headline: its NewsNow id and link, publisher and time, and the FPL players and clubs named in it. */
export interface Headline {
  id: string; title: string; publisher: string | null; t: string | null; url: string;
  players: number[]; teams: number[];
}
export interface NewsLog {
  updated: string; season: string; gw_current: number | null; gw_next: number | null;
  last_deadline: string | null; next_deadline: string | null;
  players: NewsPlayer[]; press: Press | null; press_url: string; press_name: string; log: NewsChange[];
  /** The last two weeks of headlines (absent in a file written before they were added). */
  headlines?: { fetched: string | null; items: Headline[] }; headlines_url?: string; headlines_name?: string;
}

/** Where the live file is: the `news` branch through raw.githubusercontent.com on github.io, else the copy
 * `xpfpl export` wrote (as for prices). */
export function newsUrl(): string {
  const { hostname, pathname } = window.location;
  if (!hostname.endsWith(".github.io")) return "news.json";
  const owner = hostname.slice(0, -".github.io".length), repo = pathname.split("/")[1];
  return `https://raw.githubusercontent.com/${owner}/${repo}/news/news.json`;
}

/** Each player's latest headline (the newest NewsNow headline naming him), for a source when FPL gives none. */
export function latestHeadlines(log: NewsLog | null | undefined): Map<number, Headline> {
  const out = new Map<number, Headline>();
  for (const h of log?.headlines?.items ?? []) {
    for (const id of h.players) {
      const had = out.get(id);
      if (!had || (h.t ?? "") > (had.t ?? "")) out.set(id, h);
    }
  }
  return out;
}

/** What set the chance of playing this model used (data/news.py `availability`). */
export const NEWS_RULE: Record<string, string> = {
  flag: "FPL's flag",
  press: "a press conference newer than FPL's news",
  back: "FPL's return date",
  risk: "a known absence",
};

/** A day as "Sat 10 Oct" (ISO date or time). */
export function shortDay(iso: string | null | undefined): string {
  if (!iso) return "–";
  const d = /^\d{4}-\d{2}-\d{2}$/.test(iso) ? new Date(`${iso}T12:00:00Z`) : new Date(iso);
  return d.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" });
}

/** How long ago, in words ("3 h ago", "5 days ago"). */
export function ago(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "";
  const hours = (now - Date.parse(iso)) / 3_600_000;
  if (hours < 1) return "under an hour ago";
  if (hours < 48) return `${Math.round(hours)} h ago`;
  return `${Math.round(hours / 24)} days ago`;
}
