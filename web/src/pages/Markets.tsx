import * as Plot from "@observablehq/plot";
import { useCallback, useMemo, useState } from "react";
import { color } from "../colors";
import { Chart, Club, Legend, Loading, Note, Segmented, Table, plotDefaults, useClubName, type Column } from "../components/ui";
import { rows, type Markets as MarketsFile, type Row } from "../data";
import { dec, pct, pts, signed, when } from "../format";
import { resultHistory, oddsUrl, type LiveScorer, type MatchHistory, type OddsSnapshot, type Outright } from "../polymarket";
import { matchPlayer, ours } from "../ratings";
import { useData, useSite, type Site } from "../site";

const LIVE = "live";

/** One match, whichever source it came from (the export's deadline odds or today's live odds). */
interface MatchView {
  slug: string; kickoff: string; gw: number | null; home_code: number; away_code: number;
  home_win: number; draw: number; away_win: number; over25: number | null; btts: number | null;
  cs_home: number; cs_away: number; lam_home: number; lam_away: number;
  ours_home: number | null; ours_away: number | null; ours_home_win: number | null; ours_away_win: number | null;
  goals_home: number | null; goals_away: number | null; xg_home: number | null; xg_away: number | null;
  label: string; volume: number;
}

const nan = (v: unknown): number | null => (typeof v === "number" && !Number.isNaN(v) ? v : null);

function fromExport(r: Row, club: (c: number) => string): MatchView {
  const played = r.goals_home !== null && r.goals_home !== undefined;
  const score = played ? ` ${r.goals_home}–${r.goals_away} ` : " v ";
  return {
    slug: r.slug, kickoff: r.kickoff, gw: r.gw, home_code: r.home_code, away_code: r.away_code,
    home_win: r.home_win, draw: r.draw, away_win: r.away_win, over25: nan(r["over_2.5"]), btts: nan(r.btts),
    cs_home: nan(r["away_over_0.5"]) !== null ? 1 - r["away_over_0.5"] : Math.exp(-r.lam_away),
    cs_away: nan(r["home_over_0.5"]) !== null ? 1 - r["home_over_0.5"] : Math.exp(-r.lam_home),
    lam_home: r.lam_home, lam_away: r.lam_away,
    ours_home: r.ours_home, ours_away: r.ours_away, ours_home_win: r.ours_home_win, ours_away_win: r.ours_away_win,
    goals_home: r.goals_home, goals_away: r.goals_away, xg_home: r.xg_home, xg_away: r.xg_away,
    label: `${club(r.home_code)}${score}${club(r.away_code)}`, volume: r.volume ?? 0,
  };
}

/** The FPL gameweek of an upcoming match: the fixture between the same clubs within a few days. */
function fixtureGw(site: Site, home: number, away: number, kickoff: string): number | null {
  const h = site.teamByCode.get(home)?.id, a = site.teamByCode.get(away)?.id;
  const f = site.fixtures.find((x) => x.home === h && x.away === a && x.kickoff &&
    Math.abs(new Date(x.kickoff).getTime() - new Date(kickoff).getTime()) < 5 * 86400e3);
  return f?.gw ?? null;
}

// ---------------------------------------------------------------- charts

function WinOdds({ matches }: { matches: MatchView[] }) {
  const long = useMemo(() => matches.flatMap((m, i) => [
    { label: m.label, order: i, result: "Home win", p: m.home_win },
    { label: m.label, order: i, result: "Draw", p: m.draw },
    { label: m.label, order: i, result: "Away win", p: m.away_win },
  ]), [matches]);
  const make = useCallback((width: number) => Plot.plot({
    ...plotDefaults(width),
    marginLeft: 120,
    height: Math.max(120, matches.length * 30 + 40),
    x: { label: "Market odds of each result", tickFormat: "%", domain: [0, 1] },
    y: { label: null, domain: matches.map((m) => m.label) },
    color: { domain: ["Home win", "Draw", "Away win"], range: [color.s1, color.neutral, color.s2] },
    marks: [
      Plot.barX(long, Plot.stackX({ order: ["Home win", "Draw", "Away win"], offset: "normalize", x: "p", y: "label", fill: "result", z: "result",
        insetLeft: 1, insetRight: 1, insetTop: 3, insetBottom: 3 })),
      Plot.tip(long, Plot.pointer(Plot.stackX({ order: ["Home win", "Draw", "Away win"], offset: "normalize", x: "p", y: "label", z: "result",
        title: (d: { label: string; result: string; p: number }) => `${d.label}\n${d.result}: ${pct(d.p)}` }))),
    ],
  }), [long, matches]);
  return (
    <>
      <Legend items={[{ label: "Home win", color: color.s1 }, { label: "Draw", color: color.neutral }, { label: "Away win", color: color.s2 }]} />
      <Chart make={make} height={matches.length * 30 + 40} ariaLabel="Market odds of each result, per match" />
    </>
  );
}

/** `histories`: slug -> price history, from odds.json (upcoming) or market_history/ (played). */
function Movement({ matches, end, histories }: { matches: MatchView[]; end: (m: MatchView) => Date;
                                                 histories: Record<string, MatchHistory> | null | undefined }) {
  const [slug, setSlug] = useState(matches[0]?.slug ?? "");
  const [days, setDays] = useState(3);
  const match = matches.find((m) => m.slug === slug) ?? matches[0];
  const history = match && histories ? histories[match.slug] : undefined;
  const data = useMemo(() => (history ? resultHistory(history, end(match), days) : histories === undefined ? undefined : null),
    [history, histories, match, days, end]);
  const make = useCallback((width: number) => Plot.plot({
    ...plotDefaults(width),
    height: 260,
    x: { label: null, type: "time" },
    y: { label: "Chance", tickFormat: "%", grid: true, domain: [0, 1] },
    color: { domain: ["Home win", "Draw", "Away win"], range: [color.s1, color.neutral, color.s2] },
    marks: [
      Plot.line(data ?? [], { x: "time", y: "p", stroke: "outcome", strokeWidth: 2, curve: "step-after" }),
      Plot.ruleX(data ?? [], Plot.pointerX({ x: "time", stroke: color.muted, strokeWidth: 1 })),
      Plot.tip(data ?? [], Plot.pointerX({ x: "time", y: "p",
        title: (d: { time: Date; outcome: string; p: number }) => `${d.time.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}\n${d.outcome}: ${pct(d.p, 1)}` })),
    ],
  }), [data]);
  if (!match) return null;
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>How the odds moved before the deadline</h3>
      <div className="toolbar">
        <select aria-label="Match" value={match.slug} onChange={(e) => setSlug(e.target.value)}>
          {matches.map((m) => <option key={m.slug} value={m.slug}>{m.label}</option>)}
        </select>
        <Segmented label="Window" value={days} onChange={setDays}
                   options={[{ value: 1.5 / 24, label: "90 min" }, { value: 1, label: "1 day" }, { value: 3, label: "3 days" },
                             { value: 7, label: "7 days" }, { value: 14, label: "14 days" }]} />
      </div>
      <Legend items={[{ label: "Home win", color: color.s1, kind: "line" }, { label: "Draw", color: color.neutral, kind: "line" },
                      { label: "Away win", color: color.s2, kind: "line" }]} />
      {data === undefined && <Loading />}
      {data === null && <p className="muted">No price history saved for this match.</p>}
      {data && data.length === 0 && <p className="muted">No trades in this window.</p>}
      {data && data.length > 0 && <Chart make={make} height={260} ariaLabel={`Price history for ${match.label}`} />}
      <p className="note">Polymarket prices up to {when(end(match).toISOString())}.</p>
    </div>
  );
}

// ---------------------------------------------------------------- season markets

function SeasonMarkets({ snapshot, live, fetched }: { snapshot: MarketsFile["outrights"]; live: Outright[] | null | undefined; fetched?: string }) {
  const saved = useMemo(() => rows<Outright>(snapshot as never), [snapshot]);
  const list = live && live.length ? live : saved;
  const events = useMemo(() => [...new Set(list.map((o) => o.event))].sort(), [list]);
  const [event, setEvent] = useState<string>("");
  // The title race: "champion" as a word, not "EFL Championship" or "Champions League".
  const chosen = events.includes(event) ? event : events.find((e) => /\bchampion\b/i.test(e)) ?? events[0];
  // An outcome nobody has traded sits at its opening price (often 50%): that isn't odds.
  const outcomes = list.filter((o) => o.event === chosen && !Number.isNaN(o.probability) && o.volume > 0)
    .sort((a, b) => b.probability - a.probability).slice(0, 12);
  const make = useCallback((width: number) => Plot.plot({
    ...plotDefaults(width),
    marginLeft: 150,
    height: outcomes.length * 26 + 40,
    x: { label: "Chance", tickFormat: "%", domain: [0, Math.max(0.1, ...outcomes.map((o) => o.probability))], grid: true },
    y: { label: null, domain: outcomes.map((o) => o.outcome) },
    marks: [
      Plot.barX(outcomes, { x: "probability", y: "outcome", fill: color.s1, rx2: 4, insetTop: 4, insetBottom: 4 }),
      Plot.text(outcomes, { x: "probability", y: "outcome", text: (o: Outright) => pct(o.probability), dx: 6, textAnchor: "start", fill: color.ink2 }),
      Plot.tip(outcomes, Plot.pointerY({ x: "probability", y: "outcome", title: (o: Outright) => `${o.outcome}: ${pct(o.probability, 1)}` })),
    ],
  }), [outcomes]);
  if (live === undefined && !saved.length) return <Loading />;
  if (!list.length) return <p className="muted">No season markets available.</p>;
  return (
    <div className="card">
      <div className="toolbar">
        <select aria-label="Season market" value={chosen} onChange={(e) => setEvent(e.target.value)}>
          {events.map((e) => <option key={e} value={e}>{e}</option>)}
        </select>
        <span className="muted">{live && live.length ? `Updated ${when(fetched)}` : `Saved ${when(snapshot?.snapshot)}`}</span>
      </div>
      {outcomes.length ? <Chart make={make} height={outcomes.length * 26 + 40} ariaLabel={`${chosen}: the market's chances`} />
                       : <p className="muted">Nobody has traded this market yet.</p>}
      <p className="note">The 12 likeliest outcomes that have been traded.</p>
    </div>
  );
}

// ---------------------------------------------------------------- the page

export default function MarketsPage() {
  const site = useSite();
  const club = useClubName();
  const file = useData<MarketsFile>("markets.json");
  // Live odds come from odds.json, which a scheduled GitHub Action refreshes: the browser
  // never calls Polymarket (it's blocked on some networks). Matches that have kicked off since drop out.
  const odds = useData<OddsSnapshot>(oddsUrl());
  const live = useMemo(() => {
    if (!odds) return odds;
    const upcoming = odds.matches.filter((m) => new Date(m.kickoff).getTime() > Date.now());
    const slugs = new Set(upcoming.map((m) => m.slug));
    return {
      scorers: odds.scorers.filter((s) => slugs.has(s.slug)),
      matches: upcoming.map((m): MatchView => ({
        slug: m.slug, kickoff: m.kickoff, gw: fixtureGw(site, m.home_code, m.away_code, m.kickoff),
        home_code: m.home_code, away_code: m.away_code, home_win: m.prices.home_win, draw: m.prices.draw ?? NaN,
        away_win: m.prices.away_win ?? NaN, over25: nan(m.prices["over_2.5"]), btts: nan(m.prices.btts),
        cs_home: "away_over_0.5" in m.prices ? 1 - m.prices["away_over_0.5"] : Math.exp(-m.lam_away),
        cs_away: "home_over_0.5" in m.prices ? 1 - m.prices["home_over_0.5"] : Math.exp(-m.lam_home),
        lam_home: m.lam_home, lam_away: m.lam_away, ...ours(site, m.home_code, m.away_code),
        goals_home: null, goals_away: null, xg_home: null, xg_away: null,
        label: `${club(m.home_code)} v ${club(m.away_code)}`, volume: m.volume,
      })),
    };
  }, [odds, site]); // eslint-disable-line react-hooks/exhaustive-deps

  const saved = useMemo(() => rows(file?.matches).map((r) => ({ ...fromExport(r, club), season: r.season as string })),
    [file]); // eslint-disable-line react-hooks/exhaustive-deps
  const seasons = useMemo(() => [...new Set(saved.map((m) => m.season))].sort().reverse(), [saved]);
  const [season, setSeason] = useState(site.meta.season);
  const now = Date.now();
  const gws = useMemo(() => [...new Set(saved.filter((m) => m.season === season && new Date(m.kickoff).getTime() < now)
    .map((m) => m.gw!))].sort((a, b) => b - a), [saved, season, now]);
  const [choice, setChoice] = useState<string | number | null>(null);
  const hasLive = !!live && live.matches.length > 0;
  const pick = choice ?? (hasLive && season === site.meta.season ? LIVE : gws[0] ?? null);
  const isLive = pick === LIVE;

  const matches: MatchView[] = isLive
    ? live?.matches ?? []
    : saved.filter((m) => m.season === season && m.gw === pick).sort((a, b) => a.kickoff.localeCompare(b.kickoff));

  const played = useData<Record<string, MatchHistory>>(
    !isLive && typeof pick === "number" ? `market_history/${season}/gw${String(pick).padStart(2, "0")}.json` : null);
  const histories = isLive ? odds?.history : played;

  const deadline = useCallback((m: MatchView) => {
    if (isLive) return new Date(Math.min(new Date(odds!.fetched_at).getTime(), new Date(m.kickoff).getTime()));
    const ev = season === site.meta.season ? site.meta.events.find((e) => e.id === m.gw) : undefined;
    if (ev) return new Date(ev.deadline);
    const first = Math.min(...matches.map((x) => new Date(x.kickoff).getTime()));
    return new Date(first - 90 * 60e3);
  }, [isLive, odds, season, site, matches]);

  if (file === undefined) return <Loading />;

  const matchColumns: Column<MatchView>[] = [
    { key: "match", label: "Match", value: (m) => m.label },
    ...only<MatchView>(isLive, [{ key: "ko", label: "Kick-off", value: (m: MatchView) => m.kickoff, render: (m: MatchView) => when(m.kickoff) }]),
    { key: "home", label: "Home", group: "Market", numeric: true, value: (m) => m.home_win, render: (m) => pct(m.home_win), title: "Market odds of a home win" },
    { key: "draw", label: "Draw", group: "Market", numeric: true, value: (m) => m.draw, render: (m) => pct(m.draw), title: "Market odds of a draw" },
    { key: "away", label: "Away", group: "Market", numeric: true, value: (m) => m.away_win, render: (m) => pct(m.away_win), title: "Market odds of an away win" },
    { key: "ohome", label: "Home", group: "Our model", numeric: true, value: (m) => m.ours_home_win, render: (m) => pct(m.ours_home_win),
      title: "Chance of a home win from the model's own club ratings" },
    { key: "odraw", label: "Draw", group: "Our model", numeric: true, value: oursDraw, render: (m) => pct(oursDraw(m)),
      title: "Chance of a draw from the model's own club ratings" },
    { key: "oaway", label: "Away", group: "Our model", numeric: true, value: (m) => m.ours_away_win, render: (m) => pct(m.ours_away_win),
      title: "Chance of an away win from the model's own club ratings" },
    { key: "xg", label: "Market", group: "Goals (Home – Away)", value: (m) => m.lam_home + m.lam_away,
      title: "Expected goals each side, fitted to all the goal markets", render: (m) => `${dec(m.lam_home, 1)} – ${dec(m.lam_away, 1)}` },
    { key: "oxg", label: "Ours", group: "Goals (Home – Away)", value: (m) => m.ours_home, title: "Expected goals each side from the model's own club ratings",
      render: (m) => m.ours_home === null ? "–" : `${dec(m.ours_home, 1)} – ${dec(m.ours_away, 1)}` },
    ...only<MatchView>(!isLive, [{ key: "actual", label: "Actual xG", group: "Goals (Home – Away)", value: (m: MatchView) => m.xg_home,
      title: "Expected goals (xG) each side in the match itself",
      render: (m: MatchView) => m.xg_home === null ? "–" : `${dec(m.xg_home, 1)} – ${dec(m.xg_away, 1)}` }]),
    { key: "csh", label: "CS Home", numeric: true, value: (m) => m.cs_home, render: (m) => pct(m.cs_home), title: "Clean-sheet chance, home side" },
    { key: "csa", label: "CS Away", numeric: true, value: (m) => m.cs_away, render: (m) => pct(m.cs_away), title: "Clean-sheet chance, away side" },
    { key: "o25", label: "Over 2.5", numeric: true, value: (m) => m.over25, render: (m) => pct(m.over25) },
    { key: "btts", label: "Both score", numeric: true, value: (m) => m.btts, render: (m) => pct(m.btts) },
    ...only<MatchView>(!isLive, [{ key: "fav", label: "Favourite won?", value: (m: MatchView) => { const f = favourite(m); return f === null ? null : Number(f); },
      render: (m: MatchView) => { const f = favourite(m); return f === null ? "–" : <span className={f ? "good" : "bad"}>{f ? "Yes" : "No"}</span>; } }]),
  ];

  return (
    <>
      <h2>Markets</h2>
      <p className="lede">
        What the betting markets thought, next to the model's own club ratings. For completed gameweeks the odds are as they
        were at the deadline, compared to what actually happened. Upcoming matches' odds are refreshed regularly.
      </p>
      <div className="toolbar">
        <label>
          Season
          <select value={season} onChange={(e) => { setSeason(e.target.value); setChoice(null); }}>
            {[...new Set([site.meta.season, ...seasons])].map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label>
          Gameweek
          <select value={String(pick)} onChange={(e) => setChoice(e.target.value === LIVE ? LIVE : Number(e.target.value))}>
            {season === site.meta.season && <option value={LIVE}>Next matches (live)</option>}
            {gws.map((g) => <option key={g} value={g}>GW{g} (played)</option>)}
          </select>
        </label>
        {isLive && live === undefined && <span className="muted">Loading odds…</span>}
        {isLive && odds && <span className="muted">Odds updated {when(odds.fetched_at)}</span>}
      </div>

      {isLive && live === null && <p>No live odds saved yet. Pick a played gameweek.</p>}
      {isLive && live && !live.matches.length && (
        <p>Polymarket hasn't listed the next matches yet. They usually appear about a week before kick-off.</p>
      )}
      {matches.length > 0 && (
        <>
          <div className="card">
            <h3 style={{ marginTop: 0 }}>Win odds {isLive ? "(live)" : `at the GW${pick} deadline`}</h3>
            <WinOdds matches={matches} />
          </div>
          <h3>Match by match</h3>
          <Table columns={matchColumns} data={matches} rowKey={(m) => m.slug} />
          <Note>
            The market's goals are the expected goals for each side that best fit all of a match's goal markets at once
            (result, totals, team totals, both teams to score). "Our model" and "Ours" come from the model's own club
            ratings{isLive ? ` for GW${site.meta.next_gw}` : " before that gameweek"}.
          </Note>
          <Movement key={`${season}-${pick}`} matches={matches} end={deadline} histories={histories} />
          <h3>Anytime goalscorer odds</h3>
          {isLive ? <LiveScorers scorers={live?.scorers ?? []} />
                  : <PlayedScorers scorers={rows(file?.scorers).filter((s) => s.season === season && s.gw === pick)} />}
        </>
      )}

      <h3>Season markets</h3>
      <SeasonMarkets snapshot={file?.outrights} live={odds === undefined ? undefined : odds?.outrights ?? null} fetched={odds?.fetched_at} />

      <h3>How good are the odds?</h3>
      <MarketAccuracy data={rows(file?.accuracy)} />
    </>
  );
}

/** The model's chance of a draw: whatever its home and away win chances leave. */
function oursDraw(m: MatchView): number | null {
  return m.ours_home_win === null || m.ours_away_win === null ? null : 1 - m.ours_home_win - m.ours_away_win;
}

/** `cols` when `show`, else nothing: for columns that only apply to live or to played matches. */
function only<T>(show: boolean, cols: Column<T>[]): Column<T>[] {
  return show ? cols : [];
}

function favourite(m: MatchView): boolean | null {
  if (m.goals_home === null || m.goals_away === null) return null;
  const fav = m.home_win >= m.away_win && m.home_win >= m.draw ? "h" : m.away_win >= m.draw ? "a" : "d";
  const result = m.goals_home > m.goals_away ? "h" : m.goals_home < m.goals_away ? "a" : "d";
  return fav === result;
}

function LiveScorers({ scorers }: { scorers: LiveScorer[] }) {
  const site = useSite();
  const list = useMemo(() => scorers.map((s) => ({ ...s, fpl: matchPlayer(site, s) })), [scorers, site]);
  if (!list.length) return <p className="muted">No scorer markets listed yet (they usually open a day or two before kick-off).</p>;
  type L = (typeof list)[number];
  const columns: Column<L>[] = [
    { key: "player", label: "Player", value: (s) => s.fpl?.web_name ?? s.player },
    { key: "club", label: "Club", value: (s) => s.fpl?.team, render: (s) => (s.fpl ? <Club id={s.fpl.team} /> : "–") },
    { key: "p", label: "Scores", numeric: true, value: (s) => (s.volume > 0 ? s.p : null),
      render: (s) => (s.volume > 0 ? pct(s.p) : <span className="muted" title="Opening price, never traded">untraded</span>) },
    { key: "out", label: "Market says", value: (s) => (s.p <= site.meta.out_threshold ? 1 : 0),
      render: (s) => (s.p <= site.meta.out_threshold ? <span className="tag warn">Likely out</span> : "") },
    { key: "xp", label: `xP GW${site.meta.next_gw}`, numeric: true, value: (s) => s.fpl?.forecast, render: (s) => pts(s.fpl?.forecast) },
  ];
  return (
    <>
      <Table columns={columns} data={list} sort="p" rowKey={(s) => `${s.slug}-${s.player}`} limit={30} />
      <Note>
        Odds of {pct(site.meta.out_threshold)} or less almost always mean the player has been ruled out: in 2025-26, 3 of the 62 players
        priced that low played. The model cuts that week's xP to 10% for them.
      </Note>
    </>
  );
}

function PlayedScorers({ scorers }: { scorers: Row[] }) {
  const site = useSite();
  if (!scorers.length) return <p className="muted">No scorer markets for this gameweek (they started in January 2026).</p>;
  const name = (s: Row) => site.players.find((p) => p.code === s.code)?.web_name ?? s.player;
  const columns: Column<Row>[] = [
    { key: "player", label: "Player", value: name },
    { key: "club", label: "Club", value: (s) => s.team_code, render: (s) => <Club code={s.team_code} /> },
    { key: "p", label: "Odds at deadline", numeric: true, value: (s) => (s.volume > 0 ? s.p_anytime : null),
      render: (s) => (s.volume > 0 ? pct(s.p_anytime) : <span className="muted" title="Opening price, never traded">untraded</span>) },
    { key: "move", label: "3-day move", numeric: true, value: (s) => (s.p_anytime_3d == null ? null : s.p_anytime - s.p_anytime_3d),
      render: (s) => (s.p_anytime_3d == null ? "–" : signed((s.p_anytime - s.p_anytime_3d) * 100, 0) + " pts") },
    { key: "min", label: "Mins", numeric: true, value: (s) => s.minutes, render: (s) => s.minutes ?? "–" },
    { key: "goals", label: "Scored?", value: (s) => s.goals,
      render: (s) => (s.goals == null ? "–" : s.goals > 0 ? <span className="good">Yes{s.goals > 1 ? ` (${s.goals})` : ""}</span> : "No") },
  ];
  return (
    <>
      <Table columns={columns} data={scorers} sort="p" rowKey={(s) => `${s.slug}-${s.player}`} limit={30} />
      <Note>Scorer markets open at about 50% and drift to a real price once traded, so an untraded one shows no odds.</Note>
    </>
  );
}

function MarketAccuracy({ data }: { data: Row[] }) {
  if (!data.length) return null;
  const columns: Column<Row>[] = [
    { key: "season", label: "Season", value: (r) => r.season },
    { key: "source", label: "Source", value: (r) => r.source },
    { key: "matches", label: "Matches", numeric: true, value: (r) => r.matches },
    { key: "goals", label: "Goals error", numeric: true, value: (r) => r.goals_rmse, render: (r) => dec(r.goals_rmse, 3),
      title: "RMSE of each side's expected goals against the goals scored" },
    { key: "cs", label: "Clean-sheet Brier", numeric: true, value: (r) => r.clean_sheet_brier, render: (r) => dec(r.clean_sheet_brier, 3) },
    { key: "win", label: "Win log loss", numeric: true, value: (r) => r.win_log_loss, render: (r) => dec(r.win_log_loss, 3) },
  ];
  return (
    <>
      <Table columns={columns} data={data} rowKey={(r) => `${r.season}-${r.source}`} />
      <Note>Lower is better for all three. "Market" is Polymarket at each FPL deadline; "Our ratings" is the model's club-strength fit as it stood then.</Note>
    </>
  );
}
