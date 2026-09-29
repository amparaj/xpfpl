// One played match (Past Gameweeks -> a result): where every shot came from and where it went,
// how the chances and the momentum built up, the team stats, what the odds said before the
// deadline, and every player's FPL points next to their xP and match stats. The match data is
// FotMob's, via FPL-Core-Insights (export.py `_matches`, data/matchstats.py).

import * as Plot from "@observablehq/plot";
import { useCallback, useMemo, useState } from "react";
import { color } from "../colors";
import { Chart, Club, Legend, Loading, Note, Segmented, Table, Tiles, plotDefaults, type Column } from "../components/ui";
import { gwFile, rows, type Gameweek, type GwRow, type Markets } from "../data";
import { POSITIONS, dec, int, pct, pts, signed, when } from "../format";
import { useData, useSite } from "../site";

// ---------------------------------------------------------------- the data

interface MatchFile {
  gw: number; source: string;
  fixtures: Record<string, {
    stats?: Record<string, [number | null, number | null]>;
    shots?: Record<string, unknown[]>; momentum?: Record<string, unknown[]>; players?: Record<string, unknown[]>;
  }>;
}
interface Shot {
  minute: number; added_time: number | null; is_home: boolean; element: number | null; outcome: string;
  situation: string; body_part: string; xg: number; xgot: number | null;
  start_x: number; start_y: number; goal_mouth_y: number; goal_mouth_z: number;
}
/** A shot placed for drawing: pitch metres (the home side attacks to the right), and the running xG. */
interface Placed extends Shot {
  side: "home" | "away"; name: string; club: string; x: number; y: number; t: number; label: string; goal: boolean; cumulative: number;
}
interface PlayerStats { element: number; [stat: string]: number | null }

const matchFile = (gw: number) => `matches/gw${String(gw).padStart(2, "0")}.json`;

const LENGTH = 105, WIDTH = 68;      // metres; the source's coordinates are 0-100 both ways
const OUTCOMES: Record<string, string> = {
  goal: "Goal", save: "Saved", miss: "Off target", block: "Blocked", post: "Hit the woodwork",
  "blocked-off-line": "Cleared off the line", "own-goal": "Own goal",
};
const SITUATIONS: Record<string, string> = {
  assisted: "Open play, assisted", regular: "Open play", "fast-break": "Counter-attack", corner: "From a corner",
  "set-piece": "Set piece", "throw-in-set-piece": "From a throw-in", "free-kick": "Direct free kick", penalty: "Penalty",
};
const BODY: Record<string, string> = { "right-foot": "right foot", "left-foot": "left foot", head: "header", other: "other" };

/** The minute as shown ("45+2'") and a position on the time axis that keeps first-half stoppage
 * time before the second half starts. */
function clock(s: Shot): { label: string; t: number } {
  const added = s.added_time ?? 0;
  return {
    label: added ? `${s.minute}+${added}'` : `${s.minute}'`,
    t: s.minute + (s.minute === 45 ? Math.min(added, 4) * 0.2 : added),
  };
}

// ---------------------------------------------------------------- the page

export default function Match({ gw, fixture }: { gw: number; fixture: number }) {
  const site = useSite();
  const week = useData<Gameweek>(gwFile(gw));
  const file = useData<MatchFile>(matchFile(gw));
  const markets = useData<Markets>("markets.json");
  const fx = site.fixtures.find((f) => f.id === fixture);
  const data = file?.fixtures[String(fixture)];

  const shots: Placed[] = useMemo(() => {
    if (!data?.shots || !fx) return [];
    const running = { home: 0, away: 0 };
    return rows<Shot>(data.shots).map((s) => {
      const side = s.is_home ? "home" : "away";
      const p = s.element !== null ? site.player.get(s.element) : undefined;
      const club = site.team.get(side === "home" ? fx.home : fx.away)?.short ?? "";
      running[side] += s.xg;
      // The source measures x from the goal being attacked; the home side shoots at the right-hand goal.
      const along = LENGTH * (1 - s.start_x / 100), across = WIDTH * (s.start_y / 100);
      return {
        ...s, side, club, name: p?.web_name ?? "Player not recorded", goal: s.outcome === "goal",
        x: side === "home" ? along : LENGTH - along, y: side === "home" ? across : WIDTH - across,
        ...clock(s), cumulative: running[side],
      };
    });
  }, [data, fx, site]);

  if (!fx) return <p>That match isn't in this season's fixtures. <a href={`#gameweeks/${gw}`}>Back to Gameweek {gw}</a></p>;
  const home = site.team.get(fx.home), away = site.team.get(fx.away);
  const stats = data?.stats ?? {};
  const stat = (name: string) => stats[name] ?? [null, null];
  const [xgH, xgA] = stat("expected_goals_xg");
  const [shotsH, shotsA] = stat("total_shots");
  const [onH, onA] = stat("shots_on_target");
  const [bigH, bigA] = stat("big_chances");
  const [possH, possA] = stat("possession");

  return (
    <>
      <p className="crumb"><a href={`#gameweeks/${gw}`}>← Gameweek {gw}</a></p>
      <h2 className="match-title">
        <Club id={fx.home} /> {home?.name} <span className="score">{fx.home_score ?? "–"} – {fx.away_score ?? "–"}</span> {away?.name} <Club id={fx.away} />
      </h2>
      <p className="lede">{when(fx.kickoff)} · Gameweek {gw}</p>
      {file === undefined && <Loading />}
      {file !== undefined && !data && (
        <p>No shot data for this match yet: it arrives a day or two after the match, with the next <code>xpfpl fetch</code>.</p>
      )}
      {data && (
        <Tiles tiles={[
          { label: "Expected goals", value: `${dec(xgH)} – ${dec(xgA)}`, note: "the quality of the chances each side had" },
          { label: "Shots (on target)", value: `${int(shotsH)} (${int(onH)}) – ${int(shotsA)} (${int(onA)})` },
          { label: "Big chances", value: `${int(bigH)} – ${int(bigA)}` },
          { label: "Possession", value: `${int(possH)}% – ${int(possA)}%` },
        ]} />
      )}
      {shots.length > 0 && (
        <>
          <ShotMap shots={shots} home={home?.short ?? "Home"} away={away?.short ?? "Away"} />
          <div className="grid-2">
            <OnGoal shots={shots} home={home?.short ?? "Home"} away={away?.short ?? "Away"} />
            <Timeline shots={shots} momentum={data?.momentum} home={home?.short ?? "Home"} away={away?.short ?? "Away"} />
          </div>
        </>
      )}
      {data?.stats && <TeamStats stats={stats} home={fx.home} away={fx.away} />}
      <Odds markets={markets} fixture={fixture} homeScore={fx.home_score} awayScore={fx.away_score} />
      {week && <Players week={week} fixture={fixture} players={data?.players} home={fx.home} />}
      {data && <Note>Match data: {file?.source}. Expected goals (xG) is the chance a shot like that is scored, from where and how it
        was taken; xG on target (xGOT) is the same once it's on target, knowing where it was placed. About one shot in eight
        has no player recorded by the source.</Note>}
    </>
  );
}

// ---------------------------------------------------------------- pitch

/** Pitch markings in metres, as line strings (a 105 x 68 pitch, y up). */
function pitchLines(): [number, number][][] {
  const arc = (cx: number, cy: number, r: number, from: number, to: number) =>
    Array.from({ length: 41 }, (_, i) => {
      const a = from + ((to - from) * i) / 40;
      return [cx + r * Math.cos(a), cy + r * Math.sin(a)] as [number, number];
    });
  const box = (x0: number, depth: number, width: number, dir: 1 | -1): [number, number][] => {
    const y0 = (WIDTH - width) / 2, y1 = (WIDTH + width) / 2;
    return [[x0, y0], [x0 + dir * depth, y0], [x0 + dir * depth, y1], [x0, y1]];
  };
  // The arc at the edge of each penalty area: the part of the 9.15 m circle round the spot outside the box.
  const d = Math.acos(5.5 / 9.15);
  return [
    [[0, 0], [LENGTH, 0], [LENGTH, WIDTH], [0, WIDTH], [0, 0]],
    [[LENGTH / 2, 0], [LENGTH / 2, WIDTH]],
    arc(LENGTH / 2, WIDTH / 2, 9.15, 0, 2 * Math.PI),
    box(0, 16.5, 40.32, 1), box(LENGTH, 16.5, 40.32, -1),
    box(0, 5.5, 18.32, 1), box(LENGTH, 5.5, 18.32, -1),
    arc(11, WIDTH / 2, 9.15, -d, d), arc(LENGTH - 11, WIDTH / 2, 9.15, Math.PI - d, Math.PI + d),
    [[0, WIDTH / 2 - 3.66], [-1.5, WIDTH / 2 - 3.66], [-1.5, WIDTH / 2 + 3.66], [0, WIDTH / 2 + 3.66]],
    [[LENGTH, WIDTH / 2 - 3.66], [LENGTH + 1.5, WIDTH / 2 - 3.66], [LENGTH + 1.5, WIDTH / 2 + 3.66], [LENGTH, WIDTH / 2 + 3.66]],
  ];
}

const shotTitle = (s: Placed) =>
  `${s.name} (${s.club}) · ${s.label}\n${OUTCOMES[s.outcome] ?? s.outcome} · xG ${dec(s.xg)}` +
  (s.xgot !== null ? ` · xGOT ${dec(s.xgot)}` : "") +
  `\n${SITUATIONS[s.situation] ?? s.situation}, ${BODY[s.body_part] ?? s.body_part}`;

type Filter = "both" | "home" | "away";

function ShotMap({ shots, home, away }: { shots: Placed[]; home: string; away: string }) {
  const [filter, setFilter] = useState<Filter>("both");
  const [goalsOnly, setGoalsOnly] = useState(false);
  const shown = shots.filter((s) => (filter === "both" || s.side === filter) && (!goalsOnly || s.goal));
  const make = useCallback((width: number) => {
    const w = Math.min(width, 820);
    const side = (s: Placed) => (s.side === "home" ? color.s1 : color.s2);
    // Biggest chances first, so the small ones sit on top and stay hoverable.
    const ordered = [...shown].sort((a, b) => b.xg - a.xg);
    return Plot.plot({
      ...plotDefaults(w),
      width: w, height: Math.round(((w - 16) * (WIDTH + 4)) / (LENGTH + 4)) + 16,
      marginLeft: 8, marginRight: 8, marginTop: 8, marginBottom: 8,
      x: { domain: [-2, LENGTH + 2], axis: null }, y: { domain: [-2, WIDTH + 2], axis: null },
      r: { domain: [0, 1], range: [4, 22] },
      marks: [
        Plot.rect([[0, 0]], { x1: 0, x2: LENGTH, y1: 0, y2: WIDTH, fill: color.grid, fillOpacity: 0.35 }),
        ...pitchLines().map((l) => Plot.line(l, { stroke: color.muted, strokeWidth: 1.2, strokeOpacity: 0.8 })),
        Plot.dot([[11, WIDTH / 2], [LENGTH - 11, WIDTH / 2], [LENGTH / 2, WIDTH / 2]], { r: 1.5, fill: color.muted }),
        Plot.dot(ordered.filter((s) => !s.goal), {
          x: "x", y: "y", r: "xg", fill: side, fillOpacity: 0.18, stroke: side, strokeWidth: 1.5,
        }),
        Plot.dot(ordered.filter((s) => s.goal), {
          x: "x", y: "y", r: "xg", fill: side, stroke: color.surface, strokeWidth: 2,
        }),
        // Each scorer named once (by his best chance), and not at all on a phone, where they'd collide.
        Plot.text(w < 560 ? [] : ordered.filter((s, i) => s.goal && ordered.findIndex((o) => o.goal && o.name === s.name) === i), {
          x: "x", y: "y", text: "name", dy: -14, fontSize: 11, fill: color.ink, stroke: color.surface, strokeWidth: 3,
        }),
        Plot.tip(ordered, Plot.pointer({ x: "x", y: "y", title: shotTitle, maxRadius: 30 })),
      ],
    });
  }, [shown]);
  const xg = (side: "home" | "away") => shots.filter((s) => s.side === side).reduce((t, s) => t + s.xg, 0);
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Shot map</h3>
      <div className="toolbar">
        <Segmented label="Which side's shots" value={filter} onChange={setFilter}
                   options={[{ value: "both", label: "Both" }, { value: "home", label: home }, { value: "away", label: away }]} />
        <label><input type="checkbox" checked={goalsOnly} onChange={(e) => setGoalsOnly(e.target.checked)} /> Goals only</label>
      </div>
      <Legend items={[
        { label: `${home} (attacking →) · ${dec(xg("home"))} xG from these shots`, color: color.s1, kind: "dot" },
        { label: `${away} (← attacking) · ${dec(xg("away"))} xG from these shots`, color: color.s2, kind: "dot" },
      ]} />
      <Chart make={make} height={320} ariaLabel="Shot map: where each shot was taken, sized by its expected goals" />
      <p className="note" style={{ marginTop: 0 }}>
        One circle per shot, where it was taken. The bigger the circle, the better the chance (xG). Solid circles are goals;
        hover (or tap) any circle for the player, minute, outcome and how the chance came about.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- where the shots went

function OnGoal({ shots, home, away }: { shots: Placed[]; home: string; away: string }) {
  // Shots that reached the goal: saved, scored, hit the woodwork or were cleared off the line.
  const reached = shots.filter((s) => ["goal", "save", "post", "blocked-off-line"].includes(s.outcome));
  const make = useCallback((width: number) => {
    const w = Math.min(width, 560);
    const side = (s: Placed) => (s.side === "home" ? color.s1 : color.s2);
    // As the shooter sees it: the source's goal-mouth y runs to the shooter's left.
    const across = (s: Placed) => 100 - s.goal_mouth_y;
    const frame: [number, number][] = [[44.8, 0], [44.8, 38], [55.2, 38], [55.2, 0]];
    return Plot.plot({
      ...plotDefaults(w),
      width: w, height: Math.round(w * 0.5), marginLeft: 8, marginRight: 8, marginTop: 12, marginBottom: 16,
      x: { domain: [43, 57], axis: null }, y: { domain: [-2, 44], axis: null },
      r: { domain: [0, 1], range: [4, 16] },
      marks: [
        Plot.ruleY([0], { stroke: color.muted }),
        Plot.line(frame, { stroke: color.ink2, strokeWidth: 3 }),
        Plot.dot(reached, {
          x: across, y: "goal_mouth_z", r: (s: Placed) => s.xgot ?? s.xg, fill: side,
          fillOpacity: (s: Placed) => (s.goal ? 1 : 0.18), stroke: (s: Placed) => (s.goal ? color.surface : side(s)),
          strokeWidth: (s: Placed) => (s.goal ? 2 : 1.5),
        }),
        Plot.tip(reached, Plot.pointer({ x: across, y: "goal_mouth_z", title: shotTitle, maxRadius: 24 })),
      ],
    });
  }, [reached]);
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Where the shots on target went</h3>
      <Legend items={[{ label: home, color: color.s1, kind: "dot" }, { label: away, color: color.s2, kind: "dot" }]} />
      {reached.length ? <Chart make={make} height={240} ariaLabel="Shots on target placed on the goal frame" />
        : <p className="muted">No shots on target.</p>}
      <p className="note" style={{ marginTop: 0 }}>
        Both sides' shots on the same goal, as the shooter sees it. Size is xG on target (how likely a shot placed there is to go
        in); solid circles are goals.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- how the match went

function Timeline({ shots, momentum, home, away }: {
  shots: Placed[]; momentum?: Record<string, unknown[]>; home: string; away: string;
}) {
  const [view, setView] = useState<"xg" | "momentum">("xg");
  const flow = useMemo(() => rows<{ minute: number; value: number }>(momentum), [momentum]);
  const end = Math.max(95, ...shots.map((s) => s.t + 1), ...flow.map((m) => m.minute + 1));
  const goals = shots.filter((s) => s.goal);
  const make = useCallback((width: number) => {
    const side = (s: { side: string }) => (s.side === "home" ? color.s1 : color.s2);
    if (view === "momentum") {
      const tip = (m: { minute: number; value: number }) =>
        `${m.minute}' · ${m.value > 0 ? home : m.value < 0 ? away : "Even"}${m.value ? ` on top (${Math.abs(m.value)})` : ""}`;
      return Plot.plot({
        ...plotDefaults(width), height: 240,
        x: { domain: [0, end], label: "Minute", ticks: [0, 15, 30, 45, 60, 75, 90] },
        y: { domain: [-100, 100], label: `← ${away} · ${home} →`, ticks: [] },
        marks: [
          Plot.rectY(flow, { x: "minute", interval: 1, y: "value", fill: (m) => (m.value >= 0 ? color.s1 : color.s2), inset: 0.5 }),
          Plot.ruleY([0], { stroke: color.muted }),
          Plot.ruleX([45], { stroke: color.grid, strokeDasharray: "3,3" }),
          Plot.dot(goals, { x: "t", y: (s: Placed) => (s.side === "home" ? 92 : -92), r: 5, fill: side, stroke: color.surface, strokeWidth: 2 }),
          Plot.tip(flow, Plot.pointerX({ x: "minute", y: "value", title: tip })),
        ],
      });
    }
    // Running xG: steps up at each shot, from zero at kick-off to the final total.
    const steps = (s: "home" | "away") => {
      const mine = shots.filter((x) => x.side === s);
      return [{ t: 0, cumulative: 0, side: s }, ...mine, { t: end, cumulative: mine.at(-1)?.cumulative ?? 0, side: s }];
    };
    const tip = (s: Placed) => `${s.name} (${s.club}) · ${s.label}\n${OUTCOMES[s.outcome] ?? s.outcome} · xG ${dec(s.xg)}\n${s.club} total ${dec(s.cumulative)} xG`;
    return Plot.plot({
      ...plotDefaults(width), height: 240,
      x: { domain: [0, end], label: "Minute", ticks: [0, 15, 30, 45, 60, 75, 90] },
      y: { label: "Expected goals so far", grid: true, nice: true },
      marks: [
        Plot.ruleX([45], { stroke: color.grid, strokeDasharray: "3,3" }),
        Plot.line(steps("home"), { x: "t", y: "cumulative", curve: "step-after", stroke: color.s1, strokeWidth: 2 }),
        Plot.line(steps("away"), { x: "t", y: "cumulative", curve: "step-after", stroke: color.s2, strokeWidth: 2 }),
        Plot.dot(goals, { x: "t", y: "cumulative", r: 5, fill: side, stroke: color.surface, strokeWidth: 2 }),
        Plot.text(goals, { x: "t", y: "cumulative", text: (s: Placed) => `${s.name} ${s.label}`, dy: -12, fontSize: 11,
                           fill: color.ink, stroke: color.surface, strokeWidth: 3 }),
        Plot.tip(shots, Plot.pointer({ x: "t", y: "cumulative", title: tip })),
      ],
    });
  }, [view, shots, flow, goals, end, home, away]);
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>How the match went</h3>
      <div className="toolbar">
        <Segmented label="Chart" value={view} onChange={setView}
                   options={[{ value: "xg", label: "Expected goals" }, ...(flow.length ? [{ value: "momentum" as const, label: "Momentum" }] : [])]} />
      </div>
      <Legend items={[{ label: home, color: color.s1, kind: "line" }, { label: away, color: color.s2, kind: "line" }]} />
      <Chart make={make} height={240} ariaLabel={view === "xg" ? "Expected goals built up over the match" : "Momentum by minute"} />
      <p className="note" style={{ marginTop: 0 }}>
        {view === "xg"
          ? "Each step is a shot, as big as its xG; dots are goals. Hover for the shot."
          : `FotMob's momentum: which side was on top each minute (up: ${home}, down: ${away}). Dots are goals.`}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- team stats

const TEAM_STATS: [string, string, number?][] = [
  ["possession", "Possession (%)", 0], ["field_tilt", "Field tilt (%)", 0], ["expected_goals_xg", "Expected goals (xG)", 2], ["xg_open_play", "xG from open play", 2],
  ["xg_set_play", "xG from set pieces", 2], ["xg_on_target_xgot", "xG on target (xGOT)", 2], ["total_shots", "Shots"],
  ["shots_on_target", "Shots on target"], ["big_chances", "Big chances"], ["big_chances_missed", "Big chances missed"],
  ["hit_woodwork", "Hit the woodwork"], ["touches_in_opposition_box", "Touches in the opposition box"], ["passes", "Passes"],
  ["accurate_passes_pct", "Pass accuracy (%)", 0], ["corners", "Corners"], ["keeper_saves", "Saves"],
  ["tackles_won", "Tackles won"], ["interceptions", "Interceptions"], ["clearances", "Clearances"], ["duels_won", "Duels won"],
  ["fouls_committed", "Fouls"], ["offsides", "Offsides"], ["yellow_cards", "Yellow cards"], ["red_cards", "Red cards"],
];

/** Field tilt: each side's share of the passes played in the opposition half (how much of the game
 * was played in each side's attacking half), from the source's `opposition_half` passes. */
function withTilt(stats: Record<string, [number | null, number | null]>) {
  const [h, a] = stats.opposition_half ?? [null, null];
  if (h === null || a === null || h + a === 0) return stats;
  return { ...stats, field_tilt: [(100 * h) / (h + a), (100 * a) / (h + a)] as [number, number] };
}

function TeamStats({ stats: raw, home, away }: { stats: Record<string, [number | null, number | null]>; home: number; away: number }) {
  const stats = withTilt(raw);
  const shown = TEAM_STATS.filter(([k]) => stats[k] && stats[k][0] !== null && stats[k][1] !== null);
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Team stats</h3>
      <p className="note" style={{ marginTop: 0 }}>Field tilt: each side's share of the passes played in the opposition half, i.e. how much
        of the match was played in its attacking half. Hover a row for the exact numbers.</p>
      <div className="versus">
        <div className="versus-head"><Club id={home} /><Club id={away} /></div>
        {shown.map(([key, label, digits = 0]) => {
          const [h, a] = stats[key].map((v) => v ?? 0);
          const total = h + a;
          const share = total > 0 ? h / total : 0.5;
          return (
            <div className="versus-row" key={key} title={`${label}: ${dec(h, digits)} – ${dec(a, digits)}`}>
              <span className={`versus-num${h > a ? " lead" : ""}`}>{dec(h, digits)}</span>
              <span className="versus-label">{label}</span>
              <span className={`versus-num away${a > h ? " lead" : ""}`}>{dec(a, digits)}</span>
              <span className="versus-bar">
                <span style={{ width: `${share * 100}%`, background: total ? "var(--s1)" : "var(--grid)" }} />
                <span style={{ width: `${(1 - share) * 100}%`, background: total ? "var(--s2)" : "var(--grid)" }} />
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- the odds before the deadline

function Odds({ markets, fixture, homeScore, awayScore }: {
  markets: Markets | null | undefined; fixture: number; homeScore: number | null; awayScore: number | null;
}) {
  const site = useSite();
  const m = useMemo(() => rows(markets?.matches).find((r) => r.season === site.meta.season && r.fixture === fixture), [markets, fixture, site]);
  if (!m || homeScore === null || awayScore === null) return null;
  const happened = homeScore > awayScore ? 0 : homeScore === awayScore ? 1 : 2;
  const sources = [
    { name: "Market Odds", note: "Polymarket at the FPL deadline", p: [m.home_win, m.draw, m.away_win] as number[] },
    { name: "Our Odds", note: "the model's club ratings", p: [m.ours_home_win, 1 - m.ours_home_win - m.ours_away_win, m.ours_away_win] as number[] },
  ].filter((s) => s.p.every((v) => typeof v === "number"));
  if (!sources.length) return null;
  const labels = ["Home win", "Draw", "Away win"];
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>What the odds said before the deadline</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Source</th>{labels.map((l, i) => <th key={l} className="num">{i === happened ? <strong>{l} ✓</strong> : l}</th>)}
              <th className="num" title="Minus the log of the chance given to what happened: lower is better">Log loss</th></tr>
          </thead>
          <tbody>
            {sources.map((s) => (
              <tr key={s.name}>
                <td>{s.name} <span className="muted">({s.note})</span></td>
                {s.p.map((v, i) => <td key={i} className="num">{i === happened ? <strong>{pct(v)}</strong> : pct(v)}</td>)}
                <td className="num">{dec(-Math.log(Math.max(s.p[happened], 1e-3)), 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">
        The ✓ marks what happened. Log loss scores a forecast only on the chance it gave to that result (lower is better; a third
        each scores 1.099). One match says little: see Markets for the season's scores.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- players

interface PlayerLine extends GwRow { name: string; team: number; position: number; home: boolean; s: PlayerStats | undefined }

function Players({ week, fixture, players, home }: {
  week: Gameweek; fixture: number; players?: Record<string, unknown[]>; home: number;
}) {
  const site = useSite();
  const [filter, setFilter] = useState<Filter>("both");
  const lines: PlayerLine[] = useMemo(() => {
    const stats = new Map(rows<PlayerStats>(players).map((p) => [p.element, p]));
    // xP is per gameweek, filed on the player's first match row: read it from there.
    const xp = new Map<number, number | null>();
    for (const r of rows<GwRow>(week.players)) if (!xp.has(r.element)) xp.set(r.element, r.xp);
    return rows<GwRow>(week.players).filter((r) => r.fixture === fixture && r.minutes > 0).map((r) => {
      const p = site.player.get(r.element);
      return { ...r, xp: xp.get(r.element) ?? null, name: p?.web_name ?? String(r.element), team: p?.team ?? 0,
               position: p?.element_type ?? 0, home: r.was_home, s: stats.get(r.element) };
    });
  }, [week, fixture, players, site]);
  const shown = lines.filter((l) => filter === "both" || (filter === "home") === l.home);
  const double = lines.some((l) => rows<GwRow>(week.players).filter((r) => r.element === l.element).length > 1);
  const n = (l: PlayerLine, k: string) => (l.s?.[k] ?? null) as number | null;
  const columns: Column<PlayerLine>[] = [
    { key: "name", label: "Player", value: (l) => l.name },
    { key: "team", label: "Club", value: (l) => site.team.get(l.team)?.short, render: (l) => <Club id={l.team} /> },
    { key: "pos", label: "Pos", value: (l) => l.position, render: (l) => POSITIONS[l.position] },
    { key: "minutes", label: "Mins", numeric: true, value: (l) => l.minutes, group: "FPL" },
    { key: "points", label: "Points", numeric: true, value: (l) => l.total_points, group: "FPL" },
    { key: "xp", label: "xP", numeric: true, value: (l) => l.xp, render: (l) => pts(l.xp), group: "FPL",
      title: double ? "The model's xP before the deadline, for the whole (double) gameweek" : "The model's xP before the deadline" },
    { key: "diff", label: "Points − xP", numeric: true, group: "FPL", value: (l) => (l.xp === null || double ? null : l.total_points - l.xp),
      render: (l) => (l.xp === null || double ? "–" : <span className={l.total_points >= l.xp ? "good" : "bad"}>{signed(l.total_points - l.xp)}</span>) },
    { key: "bonus", label: "Bonus", numeric: true, value: (l) => l.bonus, group: "FPL" },
    { key: "bps", label: "BPS", numeric: true, value: (l) => l.bps, group: "FPL" },
    { key: "shots", label: "Shots", numeric: true, value: (l) => n(l, "total_shots"), group: "Attack" },
    { key: "xg", label: "xG", numeric: true, value: (l) => n(l, "xg") ?? l.expected_goals, render: (l) => dec(n(l, "xg") ?? l.expected_goals), group: "Attack" },
    { key: "xa", label: "xA", numeric: true, value: (l) => n(l, "xa") ?? l.expected_assists, render: (l) => dec(n(l, "xa") ?? l.expected_assists), group: "Attack" },
    { key: "chances", label: "Chances created", numeric: true, value: (l) => n(l, "chances_created"), group: "Attack" },
    { key: "box", label: "Box touches", numeric: true, value: (l) => n(l, "touches_opposition_box"), group: "Attack",
      title: "Touches in the opposition box" },
    { key: "touches", label: "Touches", numeric: true, value: (l) => n(l, "touches"), group: "On the ball" },
    { key: "passes", label: "Pass %", numeric: true, value: (l) => n(l, "accurate_passes_percent"),
      render: (l) => (n(l, "accurate_passes_percent") === null ? "–" : `${int(n(l, "accurate_passes_percent"))}%`), group: "On the ball" },
    { key: "dc", label: "DC", numeric: true, value: (l) => l.defensive_contribution, group: "Defence",
      title: "FPL's defensive contribution: clearances, blocks, interceptions, tackles (and recoveries)" },
    { key: "tackles", label: "Tackles", numeric: true, value: (l) => n(l, "tackles"), group: "Defence" },
    { key: "saves", label: "Saves", numeric: true, value: (l) => l.saves, group: "Defence" },
    { key: "km", label: "Distance (km)", numeric: true, value: (l) => n(l, "distance_covered"),
      render: (l) => dec(n(l, "distance_covered"), 1), group: "Running" },
  ];
  if (!lines.length) return null;
  return (
    <>
      <h3>Players</h3>
      <div className="toolbar">
        <Segmented label="Which side" value={filter} onChange={setFilter} options={[
          { value: "both", label: "Both" },
          { value: "home", label: site.team.get(home)?.short ?? "Home" },
          { value: "away", label: site.team.get(lines.find((l) => !l.home)?.team ?? 0)?.short ?? "Away" },
        ]} />
      </div>
      <Table columns={columns} data={shown} sort="points" rowKey={(l) => l.element} />
      <Note>Everyone who played. FPL columns are FPL's own; the rest are the match data's.
        {double && " In a double gameweek xP covers both matches, so Points − xP is left blank."}</Note>
    </>
  );
}
