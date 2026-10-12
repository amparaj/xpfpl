import * as Plot from "@observablehq/plot";
import { useCallback, useMemo, useState, type ReactNode } from "react";
import { color } from "../colors";
import { band } from "../components/Simulation";
import { Chart, Club, Legend, Loading, Note, Segmented, Table, Tiles, plotDefaults, type Column, type TileProps } from "../components/ui";
import { rows, type Accuracy, type Gameweek } from "../data";
import { POSITIONS, dec, int, pct, pts, signed, when } from "../format";
import { useLive, type Live, type LiveEvents, type LiveMatch, type LivePick, type LivePlayer, type LiveSummary, type LiveTeam } from "../live";
import { useData, useHash, useSite } from "../site";
import Match from "./Match";

type Row = LivePlayer & { name: string; team: number; pos: number };
/** A row of the tables: a player, and where he was picked when it's a team's table. */
type PickRow = Row & { role?: LivePick["role"]; counted?: boolean; multiplier?: number };

const PARTS = [
  { key: "minutes", label: "Minutes", what: "played when he might not have (+), or didn't play (−), against the chance he was given" },
  { key: "finishing", label: "Finishing", what: "goals and assists beyond (+) or short of (−) the xG and xA of his chances in the match: mostly luck" },
  { key: "other", label: "Chances & the rest", what: "how many chances he got against what his xP expected, clean sheets, bonus, saves, defensive contribution and cards" },
] as const;

const STATE = { done: "Over", live: "Playing", "to play": "To play", blank: "No match" } as Record<string, string>;
const DAY = 24 * 3600 * 1000;

/** "3 days" / "20 hours": how long before `later` `earlier` was. */
function before(earlier: string | null | undefined, later: string | null | undefined): string | null {
  if (!earlier || !later) return null;
  const ms = Date.parse(later) - Date.parse(earlier);
  if (!(ms > 0)) return null;
  return ms >= 2 * DAY ? `${Math.round(ms / DAY)} days` : `${Math.round(ms / 3600000)} hours`;
}

/** Scored minus forecast in words: within half a point is "right on it". */
const beat = (d: number | null) => (d === null ? "" : Math.abs(d) < 0.5 ? "right on it" : `${int(Math.abs(d))} ${d > 0 ? "more" : "fewer"} than forecast`);

export default function LiveGameweek() {
  const live = useLive();
  const fixture = Number(useHash().split("/")[1]);
  if (live && fixture) {
    const m = live.matches.find((x) => x.id === fixture);
    if (m) return <LiveMatchPage live={live} match={m} />;
  }
  return <Overview live={live} />;
}

function Overview({ live }: { live: Live | null | undefined }) {
  const site = useSite();
  const accuracy = useData<Accuracy>("accuracy.json");
  const [side, setSide] = useState<"beat" | "missed" | "all">("missed");
  const [teamKey, setTeamKey] = useState<"model" | "mine">("model");

  const players: Row[] = useMemo(() => rows<LivePlayer>(live?.players).flatMap((r) => {
    const p = site.player.get(r.element);
    return p ? [{ ...r, name: p.web_name, team: p.team, pos: p.element_type }] : [];
  }), [live, site]);
  const byId = useMemo(() => new Map(players.map((r) => [r.element, r])), [players]);

  const week = live?.week;
  const partsChart = useCallback((width: number) => {
    const groups: [string, LiveSummary | undefined][] = [["All", week], ...Object.entries(week?.positions ?? {})];
    const data = groups.flatMap(([g, s]) => s && s.n ? PARTS.map((p) => ({ group: g, part: p.label, value: (s as any)[p.key] as number, n: s.n })) : []);
    return Plot.plot({
      ...plotDefaults(width),
      marginLeft: 130,
      height: 30 + groups.length * 70,
      // Room either side for the value labels at the bars' ends.
      x: { label: "Points minus xP, per player", grid: true,
           domain: (() => { const v = data.map((d) => d.value); const lo = Math.min(0, ...v), hi = Math.max(0, ...v), pad = (hi - lo) * 0.18 || 0.1;
                            return [lo - (lo < 0 ? pad : 0), hi + (hi > 0 ? pad : 0)]; })() },
      y: { label: null, domain: PARTS.map((p) => p.label) },
      fy: { label: null, domain: groups.map(([g]) => g) },
      marks: [
        Plot.ruleX([0], { stroke: color.muted }),
        Plot.barX(data, { x: "value", y: "part", fy: "group", fill: (d) => (d.value >= 0 ? color.good : color.bad), insetTop: 3, insetBottom: 3 }),
        Plot.text(data, { x: "value", y: "part", fy: "group", text: (d) => signed(d.value), dx: 4, textAnchor: "start", fill: color.ink2,
                          filter: (d) => d.value >= 0 }),
        Plot.text(data, { x: "value", y: "part", fy: "group", text: (d) => signed(d.value), dx: -4, textAnchor: "end", fill: color.ink2,
                          filter: (d) => d.value < 0 }),
        Plot.tip(data, Plot.pointerY({ x: "value", y: "part", fy: "group",
          title: (d) => `${d.group} (${d.n} players): ${d.part} ${signed(d.value)} points a player` })),
      ],
    });
  }, [week]);

  if (live === undefined) return <Loading />;
  if (live === null) {
    return (
      <>
        <h2>Live Gameweek</h2>
        <p>No gameweek is in progress yet. From the next deadline this page follows that gameweek against this model's forecast.</p>
      </>
    );
  }

  const gw = live.gw;
  const finished = live.done === live.total;
  const madeBefore = before(live.forecast_made, live.deadline);
  const model = live.teams.model, mine = live.teams.mine;
  const teamBefore = before(model?.made_at, live.deadline);
  const staleTeam = model?.made_at && live.deadline && Date.parse(live.deadline) - Date.parse(model.made_at) > 2 * DAY;

  const teamTile = (label: string, t: LiveTeam, href: string): TileProps => ({
    label, value: `${int(t.points)} pts`,
    note: <>
      against a forecast of {t.forecast === null ? "–" : int(t.forecast)}
      {t.settled ? <>: {beat(t.difference)}{t.percentile !== null && <>, better than {pct(t.percentile)} of its simulated weeks</>}</>
        : <>; {t.still_to_play} still to play ({pts(t.still_xp)} xP to come)</>}
      {" "}(before transfer hits). <a href={href}>See the team</a>
    </>,
  });
  const tiles: TileProps[] = [
    ...(model ? [teamTile("This Model's Team", model, "#model-team")] : []),
    ...(mine ? [teamTile("My Team", mine, "#my-team")] : []),
    ...(week && week.n ? [
      { label: "Players expected to feature", value: `${signed(week.bias)} a player`,
        note: `scored ${pts(week.points)} each against ${pts(week.xp)} xP (${week.n} players whose matches are over)` },
      ...(week.inside != null ? [{ label: "Inside their range", value: pct(week.inside),
        note: <>of those scores landed in the middle 80% of simulated scores: about 80% is right</> }] : []),
      ...(week.hauls != null ? [{ label: "Hauls (10+)", value: `${week.hauls} vs ${dec(week.hauls_expected, 1)}`,
        note: <>against the number the forecast's chances add up to; 2 or fewer: {week.blanks} vs {dec(week.blanks_expected, 0)}</> }] : []),
    ] : []),
  ];

  // Everyone else: the players in the teams above are in their own tables.
  const inTeams = new Set([...(model?.picks ?? []), ...(mine?.picks ?? [])].map((p) => p.element));
  const done = players.filter((r) => r.state === "done" && !inTeams.has(r.element));
  const surprises = done.filter((r) => side === "all" || (side === "beat" ? r.error > 0 : r.error < 0))
    .sort((a, b) => Math.abs(b.error) - Math.abs(a.error)).slice(0, 30);

  // One set of columns for the teams and for everyone else; a team's table adds where he was picked.
  const team = teamKey === "mine" ? mine : model;
  const subbedIn = new Set(team?.subs.map((s) => s.in)), subbedOut = new Set(team?.subs.map((s) => s.out));
  const settled = (r: Row) => r.state === "done";
  const columns = (picked: boolean): Column<PickRow>[] => [
    { key: "name", label: "Player", value: (r) => r.name,
      render: (r) => !picked ? r.name : <>{r.name}
        {r.element === team?.armband && <span className="tag" style={{ marginLeft: 6 }}>{team?.chip === "3xc" ? "TC" : "C"}</span>}
        {subbedIn.has(r.element) && <span className="tag good-tag" style={{ marginLeft: 6 }}>Sub on</span>}
        {subbedOut.has(r.element) && <span className="tag warn" style={{ marginLeft: 6 }}>Subbed off</span>}</> },
    { key: "club", label: "Club", value: (r) => site.team.get(r.team)?.short, render: (r) => <Club id={r.team} /> },
    { key: "pos", label: "Pos", value: (r) => r.pos, render: (r) => POSITIONS[r.pos] },
    ...(picked ? [{ key: "role", label: "Picked", value: (r: PickRow) => r.role ?? "" } as Column<PickRow>] : []),
    { key: "xp", label: "xP", numeric: true, value: (r) => r.xp, render: (r) => pts(r.xp), title: `This model's forecast for GW${gw}, saved before the deadline` },
    { key: "range", label: "Range", numeric: true, value: (r) => r.pts_p90, render: (r) => band(r.pts_p10, r.pts_p90),
      title: "The middle 80% of his simulated scores" },
    { key: "points", label: "Points", numeric: true, value: (r) => (picked ? r.points * (r.multiplier ?? 1) : r.points),
      render: (r) => !picked || r.counted ? <strong>{int(r.points * (picked ? r.multiplier ?? 1 : 1))}</strong> : <span className="muted">{int(r.points)}</span>,
      title: picked ? "As counted: captain doubled, the bench not counted (unless subbed on or Bench Boost)" : undefined },
    { key: "mins", label: "Mins", numeric: true, value: (r) => r.minutes,
      render: (r) => settled(r) || r.minutes ? int(r.minutes) : <span className="muted">{STATE[r.state]}</span> },
    { key: "diff", label: "Diff", numeric: true, value: (r) => (settled(r) ? r.error : null),
      render: (r) => settled(r) ? <span className={r.error >= 0 ? "good" : "bad"}>{signed(r.error, 1)}</span> : null,
      title: "His points minus his xP (before the armband), once his matches are over" },
    { key: "verdict", label: "Mainly", value: (r) => r.verdict ?? "",
      render: (r) => r.verdict ? <span className={`tag${r.verdict === "News changed" ? " warn" : ""}`}>{r.verdict}</span> : null,
      title: "The biggest of the three parts of the difference (see below)" },
    { key: "why", label: "Why", wrap: true, value: (r) => r.reasons.join("; "), render: (r) => r.reasons.join(" · ") },
  ];

  const detail = (r: PickRow) => settled(r) ? (
    <div className="card">
      <strong>Points minus xP: {signed(r.error)}</strong>
      <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
        {PARTS.map((p) => <li key={p.key}>{p.label}: {signed((r as any)[p.key === "minutes" ? "minutes_part" : p.key])} <span className="muted">({p.what})</span></li>)}
      </ul>
      {r.reasons.length > 0 && <p className="note">{r.reasons.join(". ")}.</p>}
    </div>
  ) : <p className="muted">{r.state === "live" ? "His match is still on." : "He hasn't played yet."}</p>;

  const picks: PickRow[] = (team?.picks ?? []).flatMap((p) => {
    const r = byId.get(p.element);
    return r ? [{ ...r, role: p.role, counted: p.counted, multiplier: p.multiplier, points: p.points, minutes: p.minutes, state: p.state }] : [];
  });

  const name = (e: number) => byId.get(e)?.name ?? site.player.get(e)?.web_name ?? String(e);
  const lessonItems: { key: keyof typeof live.lessons; title: string; text: ReactNode; show: (r: Row | undefined) => string }[] = [
    { key: "backed", title: "The chances back it up",
      text: "Beat their xP by 3+ with good chances (xG + xA of 0.6 or more). Worth a look: this model sees one match of it, so their next xP rises only a little.",
      show: (r) => r ? `${int(r.points)} pts, ${dec(r.expected_goals + r.expected_assists)} xG+xA` : "" },
    { key: "lucky", title: "Returns from little",
      text: "Beat their xP by 3+ mostly through finishing, from under 0.3 xG + xA. Don't chase on this alone.",
      show: (r) => r ? `${int(r.points)} pts, ${dec(r.expected_goals + r.expected_assists)} xG+xA` : "" },
    { key: "unlucky", title: "Chances without the points",
      text: "0.7+ xG + xA in 60+ minutes and 3 points or fewer: the underlying numbers were there.",
      show: (r) => r ? `${int(r.points)} pts, ${dec(r.expected_goals + r.expected_assists)} xG+xA` : "" },
    { key: "minutes", title: "Watch their place",
      text: "Fit at the deadline and given 80%+ to play, but didn't start. Check the team news before picking them.",
      show: (r) => r ? `${int(r.minutes)} mins` : "" },
    { key: "stale", title: "News changed after the forecast",
      text: "Their fitness flag at the deadline wasn't the one this forecast was made with. Re-running the forecast nearer the deadline fixes this.",
      show: (r) => r?.reasons[0] ?? "" },
  ];

  const history = ((accuracy?.scorecard?.diagnosis ?? []) as any[]).filter((d) => d.model === live.model).sort((a, b) => a.gw - b.gw);

  return (
    <>
      <h2 className="live-title">Live Gameweek: GW{gw} <StatusBadge live={live} /></h2>
      <p className="lede">
        GW{gw} as it happens, against this {live.model} model's forecast saved before the deadline ({when(live.deadline)}):
        who beat it or fell short, and why. {live.done} of {live.total} matches over{finished ? "" : " so far"}; updated {when(live.generated)}.
        The <a href="#next">next gameweek</a> is planned on its own page.
      </p>
      {(madeBefore || staleTeam) && (
        <p className="note">
          {madeBefore && <>This forecast was saved {madeBefore} before the deadline. </>}
          {staleTeam && <><span className="tag warn"><a href="#model-team">This Model's Team</a> was picked {teamBefore} before the deadline</span>
            {" "}and not picked again since, so it missed any team news in between.</>}
        </p>
      )}

      <div className="match-strip">
        {live.matches.map((m) => (
          <a key={m.id} href={`#live/${m.id}`} className={`match-chip ${m.state.replace(" ", "-")}`}
             title={m.state === "to play" ? when(m.kickoff) : `${STATE[m.state]}: open the match`}>
            <Club id={m.home} /> {m.state === "to play" ? "v" : `${m.home_score ?? 0}–${m.away_score ?? 0}`} <Club id={m.away} />
            {m.state === "live" && <span className="tag warn">{m.minutes ?? 0}'</span>}
          </a>
        ))}
      </div>

      {tiles.length > 0 && <Tiles tiles={tiles} />}

      {(model || mine) && (
        <>
          <h3>The teams, live</h3>
          {model && mine && (
            <div className="toolbar">
              <Segmented label="Team" value={teamKey} onChange={setTeamKey}
                         options={[{ value: "model", label: "This Model's Team" }, { value: "mine", label: "My Team" }]} />
            </div>
          )}
          {team && <>
            <Table columns={columns(true)} data={picks} rowKey={(r) => r.element} cardSub={["club", "pos", "role", "verdict"]}
                   cardStats={["xp", "points", "mins"]} detail={detail} />
            <Note>
              {team.subs.length > 0 && <>Automatic substitutions so far: {team.subs.map((s) => `${name(s.in)} for ${name(s.out)}`).join(", ")}. </>}
              {team.armband !== team.captain && team.armband !== null && <>The captain didn't play, so {name(team.armband)} has the armband. </>}
              Substitutions and the vice-captain apply once a player's matches are over without him playing, as FPL does.
              {team.spread.p10 != null && <> Before the deadline, 4 simulated weeks in 5 scored {band(team.spread.p10, team.spread.p90)}.</>}
            </Note>
          </>}
        </>
      )}

      {week && week.n > 0 && (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Where the misses came from</h3>
          <Legend items={[{ label: "More points than forecast", color: color.good }, { label: "Fewer", color: color.bad }]} />
          <Chart make={partsChart} height={30 + (1 + Object.keys(week.positions ?? {}).length) * 70}
                 ariaLabel="Points minus xP per player, split into minutes, finishing and the rest, overall and by position" />
          <p className="note">
            Every player expected to feature whose matches are over: points minus xP, per player, split into three parts that add up to it.
            {PARTS.map((p) => <span key={p.key}> <strong>{p.label}</strong>: {p.what}.</span>)}
            {" "}Finishing evens out over a season. A minutes or "rest" part that stays on one side week after week is what to fix in this model.
          </p>
        </div>
      )}

      <h3>Who beat the forecast, and who fell short</h3>
      <div className="toolbar">
        <Segmented label="Show" value={side} onChange={setSide}
                   options={[{ value: "missed", label: "Fell short" }, { value: "beat", label: "Beat it" }, { value: "all", label: "Biggest either way" }]} />
      </div>
      {surprises.length ? (
        <Table columns={columns(false)} data={surprises} rowKey={(r) => r.element} sort="diff" desc={side !== "missed"}
               cardSub={["club", "pos", "verdict"]} cardStats={["xp", "points", "diff"]} detail={detail} />
      ) : <p className="muted">No matches are over yet.</p>}
      <Note>
        The 30 biggest differences among everyone else whose matches are over{inTeams.size > 0 && ": players in the teams above are in their own tables"}.
        Click a player for the split of his difference.
      </Note>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Before the next deadline</h3>
        {lessonItems.filter((l) => live.lessons[l.key]?.length).map((l) => (
          <div key={l.key} style={{ marginBottom: 12 }}>
            <strong>{l.title}</strong>
            <p className="note" style={{ marginTop: 2 }}>{l.text}</p>
            <div className="chips">
              {live.lessons[l.key].map((e) => {
                const r = byId.get(e), p = site.player.get(e);
                return (
                  <span key={e} className="lesson-chip">
                    {p && <Club id={p.team} />} {name(e)} <span className="muted">{l.show(r)}</span>
                  </span>
                );
              })}
            </div>
          </div>
        ))}
        {!Object.values(live.lessons).some((l) => l.length) && <p className="muted">Nothing stands out yet.</p>}
        {staleTeam && <p className="note">This Model's Team: run <code>xpfpl recommend</code> on deadline day, so its picks see the latest team news.</p>}
      </div>

      {history.length > 0 && (
        <>
          <h3>Earlier gameweeks</h3>
          <Table<any> columns={[
            { key: "gw", label: "GW", value: (d) => d.gw, sortable: true },
            { key: "n", label: "Players", numeric: true, value: (d) => d.n, render: (d) => int(d.n) },
            { key: "error", label: "Points − xP", numeric: true, value: (d) => d.error, render: (d) => signed(d.error) },
            ...PARTS.map((p): Column<any> => ({ key: p.key, label: p.label, numeric: true, value: (d) => d[p.key], render: (d) => signed(d[p.key]) })),
          ]} data={history} rowKey={(d) => d.gw} sort="gw" desc={false} cardTitle={(d) => `GW${d.gw}`} cardStats={["error", "minutes", "finishing", "other"]} />
          <Note>
            The same split for every gameweek played so far, per player getting minutes (from <a href="#accuracy">the live record</a>).
            One week is mostly noise: what matters is a part that stays on one side.
          </Note>
        </>
      )}
    </>
  );
}

// ---------------------------------------------------------------- one match

const EVENT_ROWS: { key: keyof LiveEvents; label: string; count?: boolean }[] = [
  { key: "goals_scored", label: "Goals", count: true }, { key: "assists", label: "Assists", count: true },
  { key: "own_goals", label: "Own goals", count: true }, { key: "penalties_missed", label: "Penalties missed", count: true },
  { key: "penalties_saved", label: "Penalties saved", count: true }, { key: "yellow_cards", label: "Yellow cards" },
  { key: "red_cards", label: "Red cards" }, { key: "saves", label: "Saves", count: true }, { key: "bonus", label: "Bonus", count: true },
];

/** A match of the gameweek in progress: FPL's live events, then the match page (shot map, xG and the rest) once
 * `xpfpl fetch` has its match data, with each player's points against his xP from live.json. */
function LiveMatchPage({ live, match }: { live: Live; match: LiveMatch }) {
  const site = useSite();
  const players = useMemo(() => rows<LivePlayer>(live.players), [live]);
  // The match page's players table reads a gameweek file: build one from live.json for this match's two clubs.
  const week: Gameweek = useMemo(() => {
    const here = players.filter((r) => (r.team === match.home || r.team === match.away) && r.minutes > 0);
    const col = <K extends string>(k: K, f: (r: LivePlayer) => unknown) => [k, here.map(f)] as const;
    return {
      gw: live.gw, xp_source: "live", fixtures: [],
      players: Object.fromEntries([
        col("element", (r) => r.element), col("fixture", () => match.id), col("was_home", (r) => r.team === match.home),
        col("minutes", (r) => r.minutes), col("total_points", (r) => r.points), col("goals_scored", (r) => r.goals_scored),
        col("assists", (r) => r.assists), col("bonus", (r) => r.bonus), col("bps", (r) => r.bps ?? null),
        col("saves", (r) => r.saves ?? null), col("defensive_contribution", (r) => r.defensive_contribution ?? null),
        col("expected_goals", (r) => r.expected_goals), col("expected_assists", (r) => r.expected_assists), col("xp", (r) => r.xp),
      ]),
    };
  }, [players, match, live.gw]);
  const name = (e: number) => site.player.get(e)?.web_name ?? String(e);
  const ev = match.events ?? {};
  const shown = EVENT_ROWS.filter((r) => (ev[r.key] as any)?.h?.length || (ev[r.key] as any)?.a?.length);
  const list = (pairs: [number, number][] | undefined, count?: boolean) =>
    (pairs ?? []).map(([e, v]) => (count && v > 1 ? `${name(e)} ${v}` : name(e))).join(", ") || "–";
  return (
    <Match gw={live.gw} fixture={match.id} back={{ href: "#live", label: `← Live Gameweek ${live.gw}` }}
           score={match.state === "to play" ? [null, null] : [match.home_score, match.away_score]} week={week}>
      <div className="card">
        <h3 style={{ marginTop: 0 }}>
          {match.state === "live" ? <>Live: {match.minutes ?? 0}' <span className="tag warn">playing</span></>
            : match.state === "done" ? "Full time" : `Kicks off ${when(match.kickoff)}`}
        </h3>
        {shown.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead><tr><th /><th><Club id={match.home} /></th><th><Club id={match.away} /></th></tr></thead>
              <tbody>
                {shown.map((r) => (
                  <tr key={r.key}>
                    <td>{r.label}{r.key === "bonus" && ev.bonus_provisional && <span className="muted"> (provisional)</span>}</td>
                    <td>{list((ev[r.key] as any)?.h, r.count)}</td>
                    <td>{list((ev[r.key] as any)?.a, r.count)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="muted" style={{ margin: 0 }}>{match.state === "to play" ? "Not started." : "Nothing yet."}</p>}
        {ev.bps && ev.bps.length > 0 && (
          <p className="note">BPS leaders: {ev.bps.map(([e, v]) => `${name(e)} ${v}`).join(", ")}.
            {ev.bonus_provisional && " Bonus is worked out from the BPS until FPL adds it (3, 2 and 1 for the top three; ties share)."}</p>
        )}
        <p className="note">FPL's own match events, as of {when(live.generated)}. The players table below has each player's points against
          his xP; the shot map and match stats follow once the match data is published, a few hours to a day after the match.</p>
      </div>
    </Match>
  );
}

// ---------------------------------------------------------------- where the gameweek is

/** Red while a match is on, amber between matches and while FPL adds the bonus and checks the data, green once it's done. */
function StatusBadge({ live }: { live: Live }) {
  const playing = live.matches.filter((m) => m.state === "live").length;
  const next = live.matches.find((m) => m.state === "to play");
  const [cls, label, title] =
    playing ? ["live", "Live", `${playing} match${playing > 1 ? "es" : ""} being played now`]
    : next ? ["pending", "In progress", `${live.done} of ${live.total} matches over; next kick-off ${when(next.kickoff)}`]
    : live.phase === "complete" ? ["complete", "Complete", "Every match is over and FPL has confirmed the points"]
    : live.phase === "checking" ? ["pending", "Checking", "Every match is over; FPL is confirming the points"]
    : ["pending", "Bonus to come", "Every match is over; FPL is still adding the bonus"];
  return <span className={`status-badge ${cls}`} title={title}><span className="status-dot" />{label}</span>;
}
