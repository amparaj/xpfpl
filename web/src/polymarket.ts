// Live Polymarket odds, fetched by the browser. Polymarket's Gamma and CLOB APIs allow requests
// from any website (FPL's API doesn't, which is why everything else comes from the export).
// A port of the parts of src/xpfpl/data/markets.py the live view needs: finding the EPL match
// events, reading each market's question, and fitting the expected goals the odds imply.

const GAMMA = "https://gamma-api.polymarket.com";
const CLOB = "https://clob.polymarket.com";
const EPL_TAG = 306;
const MAIN_SLUG = /^epl-[a-z]+-[a-z]+-\d{4}-\d{2}-\d{2}/;
const WIN = /^Will (.+?) win on /i;
const TEAM_TOTAL = /: (.+?) O\/U (\d+\.5)$/;
const TOTAL = /(?<!1st Half )(?<!2nd Half )O\/U (\d+\.5)$/;
export const TOTAL_LINES = [1.5, 2.5, 3.5];
export const TEAM_LINES = [0.5, 1.5];

interface GammaTeam { name: string; alias?: string; ordering?: "home" | "away" }
interface GammaMarket {
  question?: string; sportsMarketType?: string; clobTokenIds?: string; outcomePrices?: string;
  bestBid?: number | string; bestAsk?: number | string; volume?: string | number; closed?: boolean;
  groupItemTitle?: string;
}
interface GammaEvent {
  slug: string; title?: string; startTime?: string; endDate?: string; volume?: number | string;
  teams?: GammaTeam[]; markets?: GammaMarket[];
}

export interface LiveMatch {
  slug: string; kickoff: string; home_code: number; away_code: number; volume: number;
  prices: Record<string, number>;          // home_win, draw, away_win, over_2.5, btts, home_over_0.5...
  tokens: Record<string, string>;          // the same names -> CLOB token id (for price history)
  lam_home: number; lam_away: number;      // expected goals fitted to all the prices
}
export interface LiveScorer { slug: string; player: string; p: number; volume: number; home_code: number; away_code: number }

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} from ${url}`);
  return r.json();
}

async function events(params: Record<string, string | number>): Promise<GammaEvent[]> {
  const out: GammaEvent[] = [];
  for (let offset = 0; ; offset += 100) {
    const q = new URLSearchParams({ ...Object.fromEntries(Object.entries(params).map(([k, v]) => [k, String(v)])),
                                    limit: "100", offset: String(offset) });
    const page = await getJson<GammaEvent[]>(`${GAMMA}/events?${q}`);
    out.push(...page);
    if (page.length < 100) return out;
  }
}

const strip = (s: string) => s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z ]/g, "").trim();

/** Similarity of two strings (0-1) by shared letter pairs: stands in for Python's SequenceMatcher. */
function similarity(a: string, b: string): number {
  if (a === b) return 1;
  const pairs = (s: string) => { const m = new Map<string, number>(); for (let i = 0; i < s.length - 1; i++) { const p = s.slice(i, i + 2); m.set(p, (m.get(p) ?? 0) + 1); } return m; };
  const pa = pairs(a), pb = pairs(b);
  let shared = 0;
  for (const [p, n] of pa) shared += Math.min(n, pb.get(p) ?? 0);
  const total = Math.max(a.length - 1, 0) + Math.max(b.length - 1, 0);
  return total ? (2 * shared) / total : 0;
}

/** Which side of the event ("home"/"away") a club named in a question is. */
function which(text: string, event: GammaEvent): "home" | "away" | null {
  const target = strip(text);
  let best = 0, side: "home" | "away" | null = null;
  for (const team of event.teams ?? []) {
    const variants = [team.name, team.alias ?? "", team.name.replace(" FC", "").replace("AFC ", "")].filter(Boolean);
    const score = Math.max(...variants.map((v) => similarity(target, strip(v))));
    if (score > best) { best = score; side = team.ordering ?? null; }
  }
  return best >= 0.6 ? side : null;
}

function matchMarkets(group: GammaEvent[]): Record<string, GammaMarket> {
  const keep: Record<string, GammaMarket> = {};
  for (const event of group) {
    for (const m of event.markets ?? []) {
      const q = (m.question ?? "").trim();
      const kind = m.sportsMarketType ?? "";
      if (q.toLowerCase().includes("half") || kind.includes("half") || kind.toLowerCase().includes("spread")) continue;
      let name: string | null = null;
      const win = WIN.exec(q), teamTotal = TEAM_TOTAL.exec(q), total = TOTAL.exec(q);
      if (q.toLowerCase().endsWith("end in a draw?")) name = "draw";
      else if (win) { const s = which(win[1], event); name = s ? `${s}_win` : null; }
      else if (q.endsWith("Both Teams to Score")) name = "btts";
      else if (teamTotal && !teamTotal[1].includes(" vs. ")) {
        const s = which(teamTotal[1], event), line = Number(teamTotal[2]);
        name = s && TEAM_LINES.includes(line) ? `${s}_over_${teamTotal[2]}` : null;
      } else if (total) {
        name = TOTAL_LINES.includes(Number(total[1])) ? `over_${total[1]}` : null;
      }
      if (name) keep[name] = m;
    }
  }
  return keep;
}

/** Probability of "Yes": the order-book midpoint when there's a tight book, else the last price. */
function yesPrice(m: GammaMarket): number {
  const bid = m.bestBid === undefined ? NaN : Number(m.bestBid), ask = m.bestAsk === undefined ? NaN : Number(m.bestAsk);
  if (!Number.isNaN(bid) && !Number.isNaN(ask) && ask - bid > 0 && ask - bid < 0.2) return (bid + ask) / 2;
  const prices = JSON.parse(m.outcomePrices || "[]");
  return prices.length ? Number(prices[0]) : NaN;
}

const firstToken = (m: GammaMarket): string | null => { const t = JSON.parse(m.clobTokenIds || "[]"); return t.length ? t[0] : null; };

// ---------------------------------------------------------------- odds -> expected goals

const MAX_GOALS = 10;
const LOG_FACT = Array.from({ length: MAX_GOALS + 1 }, (_, k) => { let s = 0; for (let i = 2; i <= k; i++) s += Math.log(i); return s; });
export const poisson = (lam: number) => LOG_FACT.map((lf, k) => Math.exp(k * Math.log(Math.max(lam, 1e-9)) - lam - lf));

/** The prices the model implies for two expected-goals rates (same names as the markets). */
export function implied(lh: number, la: number): Record<string, number> {
  const ph = poisson(lh), pa = poisson(la);
  let home = 0, draw = 0, away = 0;
  const over: Record<number, number> = Object.fromEntries(TOTAL_LINES.map((x) => [x, 0]));
  for (let i = 0; i <= MAX_GOALS; i++) for (let j = 0; j <= MAX_GOALS; j++) {
    const p = ph[i] * pa[j];
    if (i > j) home += p; else if (i === j) draw += p; else away += p;
    for (const x of TOTAL_LINES) if (i + j > x) over[x] += p;
  }
  const out: Record<string, number> = { home_win: home, draw, away_win: away, btts: (1 - ph[0]) * (1 - pa[0]) };
  for (const x of TOTAL_LINES) out[`over_${x}`] = over[x];
  for (const x of TEAM_LINES) {
    out[`home_over_${x}`] = 1 - ph.slice(0, Math.floor(x) + 1).reduce((a, b) => a + b, 0);
    out[`away_over_${x}`] = 1 - pa.slice(0, Math.floor(x) + 1).reduce((a, b) => a + b, 0);
  }
  return out;
}

/** Each side's expected goals, chosen so the implied prices match the market's as closely as
 * possible (squared error over whichever markets exist): a coarse grid, then a fine one. */
export function fitRates(prices: Record<string, number>): [number, number] {
  const names = Object.keys(prices).filter((k) => !Number.isNaN(prices[k]));
  const loss = (lh: number, la: number) => { const p = implied(lh, la); return names.reduce((s, k) => s + (p[k] - prices[k]) ** 2, 0); };
  let best: [number, number] = [1.35, 1.35], bestLoss = Infinity;
  const search = (lo: [number, number], hi: [number, number], step: number) => {
    for (let lh = lo[0]; lh <= hi[0]; lh += step) for (let la = lo[1]; la <= hi[1]; la += step) {
      const l = loss(lh, la);
      if (l < bestLoss) { bestLoss = l; best = [lh, la]; }
    }
  };
  search([0.1, 0.1], [4, 4], 0.1);
  const [h, a] = best;
  search([Math.max(0.05, h - 0.1), Math.max(0.05, a - 0.1)], [h + 0.1, a + 0.1], 0.01);
  return best;
}

// ---------------------------------------------------------------- the live view

/** Every EPL match Polymarket lists that hasn't kicked off: prices, fitted goals, scorer odds. */
export async function liveOdds(teamCodes: Record<string, number>): Promise<{ matches: LiveMatch[]; scorers: LiveScorer[] }> {
  const since = new Date(Date.now() - 86400e3).toISOString().slice(0, 19) + "Z";
  const all = await events({ tag_id: EPL_TAG, end_date_min: since });
  const games = new Map<string, GammaEvent[]>();
  for (const e of all) {
    const base = MAIN_SLUG.exec(e.slug ?? "");
    const teams = e.teams ?? [];
    if (base && teams.length === 2 && teams.every((t) => t.name in teamCodes)) {
      games.set(base[0], [...(games.get(base[0]) ?? []), e]);
    }
  }
  const now = Date.now();
  const matches: LiveMatch[] = [], scorers: LiveScorer[] = [];
  for (const [slug, group] of games) {
    const main = group.find((e) => e.slug === slug);
    if (!main?.startTime || new Date(main.startTime).getTime() <= now) continue;
    const home = main.teams!.find((t) => t.ordering === "home") ?? main.teams![0];
    const away = main.teams!.find((t) => t !== home)!;
    const prices: Record<string, number> = {}, tokens: Record<string, string> = {};
    for (const [name, m] of Object.entries(matchMarkets(group))) {
      prices[name] = yesPrice(m);
      const t = firstToken(m);
      if (t) tokens[name] = t;
    }
    if (!("home_win" in prices)) continue;
    const [lam_home, lam_away] = fitRates(prices);
    const codes = { home_code: teamCodes[home.name], away_code: teamCodes[away.name] };
    matches.push({ slug, kickoff: main.startTime, ...codes,
                   volume: group.reduce((s, e) => s + Number(e.volume ?? 0), 0), prices, tokens, lam_home, lam_away });
    for (const e of group) for (const m of e.markets ?? []) {
      if (m.sportsMarketType === "soccer_anytime_goalscorer" && !m.closed) {
        scorers.push({ slug, player: (m.question ?? "").split(":")[0].trim(), p: yesPrice(m),
                       volume: Number(m.volume ?? 0), ...codes });
      }
    }
  }
  matches.sort((a, b) => a.kickoff.localeCompare(b.kickoff));
  return { matches, scorers };
}

/** Result prices over the `days` before `end`, for the movement chart. Works for played matches too. */
export async function resultHistory(slug: string, end: Date, days: number): Promise<{ time: Date; outcome: string; p: number }[]> {
  const [main] = await getJson<GammaEvent[]>(`${GAMMA}/events?slug=${encodeURIComponent(slug)}`);
  if (!main) return [];
  const labels: Record<string, string> = { home_win: "Home win", draw: "Draw", away_win: "Away win" };
  const out: { time: Date; outcome: string; p: number }[] = [];
  const stop = Math.floor(end.getTime() / 1000);
  await Promise.all(Object.entries(matchMarkets([main])).filter(([n]) => n in labels).map(async ([name, m]) => {
    const token = firstToken(m);
    if (!token) return;
    const q = new URLSearchParams({ market: token, startTs: String(stop - Math.round(days * 86400)), endTs: String(stop),
                                    fidelity: days >= 1 ? "60" : "5" });
    const { history = [] } = await getJson<{ history?: { t: number; p: number }[] }>(`${CLOB}/prices-history?${q}`);
    for (const pt of history) out.push({ time: new Date(pt.t * 1000), outcome: labels[name], p: pt.p });
  }));
  return out.sort((a, b) => a.time.getTime() - b.time.getTime());
}

export interface Outright { event: string; outcome: string; probability: number; volume: number }

/** Today's season-long markets (title, top four, relegation, top scorer...). */
export async function liveOutrights(): Promise<Outright[]> {
  const all = await events({ tag_id: EPL_TAG, closed: "false" });
  const out: Outright[] = [];
  for (const e of all) {
    if ((e.title ?? "").includes(" vs. ")) continue;
    for (const m of e.markets ?? []) {
      if (m.closed) continue;
      out.push({ event: e.title ?? "", outcome: m.groupItemTitle || m.question || "", probability: yesPrice(m),
                 volume: Number(m.volume ?? 0) });
    }
  }
  return out;
}

/** P(scoring more than the opponent) with independent Poisson goals. */
export function poissonWin(gf: number, ga: number): number {
  const pf = poisson(gf), pa = poisson(ga);
  let p = 0;
  for (let i = 0; i <= MAX_GOALS; i++) for (let j = 0; j < i; j++) p += pf[i] * pa[j];
  return p;
}
