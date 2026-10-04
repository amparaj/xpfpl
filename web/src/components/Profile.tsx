// A player's profile (profiles.json, spatial.py), shaped by his position, for this season or an
// earlier one (seasons/<season>/profiles.json):
//   GKP  the shots on target he faced on the goal frame, saves and goals conceded match by match
//   DEF  FPL's defensive contribution match by match against its target, then how far he gets
//        forward (shot maps only for a defender who shoots)
//   MID  his shots, chances created match by match and defensive contribution against its target
//   FWD  his shots and chances created
// Tiles compare him with his position's average, and the similar players are compared on his
// position's own metrics. There are no heatmaps in the source: shot locations, per-match counts
// and zone counts (touches in the box, final-third passes) are the data there is.

import * as Plot from "@observablehq/plot";
import { useCallback, useMemo, useState } from "react";
import { color } from "../colors";
import type { Player } from "../data";
import { rows } from "../data";
import { POSITIONS, dec, money, pct, pts } from "../format";
import { useData, useSeason, useSite } from "../site";
import { Chart, Club, Legend, Loading, Note, Segmented, Tiles, barPadding, plotDefaults } from "./ui";

interface Profiles {
  zones: [number, number, number, number, number][];      // zone, x0, x1 (from the goal line), y0, y1 (across)
  min_minutes: number; min_shots: number;
  dc_threshold?: Record<string, number>;
  players: Record<string, unknown[]>;
  averages: Record<string, Record<string, number | null>>;
  similar: Record<string, [number, number][]>;
  shots: Record<string, unknown[]>;
  faced?: Record<string, unknown[]>;
  games?: Record<string, unknown[]>;
}
type Num = number | null | undefined;
interface Metrics {
  element: number; minutes: number; matches: number; shots: number; xg: number; goals: number;
  touches_p90: Num; box_touches_p90: Num; final_third_passes_p90: Num; chances_created_p90: Num; shots_p90: Num;
  box_shot_share: Num; shot_distance: Num; xg_per_shot: Num; header_share: Num;
  xa_p90: Num; crosses_p90: Num; dribbles_p90: Num; tackles_p90: Num; interceptions_p90: Num; clearances_p90: Num;
  blocks_p90: Num; recoveries_p90: Num; aerials_won_p90: Num; saves_p90: Num; conceded_p90: Num; xgot_faced_p90: Num;
  goals_prevented_p90: Num; goals_prevented: Num; save_share: Num; dc_p90: Num; dc_hits: Num;
}
interface Shot {
  element: number; gw: number; fixture: number; minute: number; start_x: number; start_y: number; xg: number;
  outcome: string; situation: string; body_part: string;
}
interface Faced {
  element: number; shooter: number | null; gw: number; fixture: number; minute: number;
  goal_mouth_y: number; goal_mouth_z: number; xg: number; xgot: number | null; outcome: string;
}
interface Game {
  element: number; gw: number; fixture: number; minutes: number; dc: Num; cbi: Num; saves: Num; conceded: Num;
  xgc: Num; chances_created: Num; xa: Num; xgot_faced: Num; goals_prevented: Num;
}
type View = "zones" | "shots" | "faced" | "keeping" | "dc" | "creating";

const OUTCOMES: Record<string, string> = {
  goal: "Goal", save: "Saved", miss: "Off target", block: "Blocked", post: "Hit the woodwork", "blocked-off-line": "Cleared off the line",
};
const VIEW_LABELS: Record<View, string> = {
  zones: "Shot zones", shots: "Every shot", faced: "Shots faced", keeping: "Match by match", dc: "Defensive contribution",
  creating: "Chances created",
};
// Half a 105 x 68 pitch, drawn with the goal at the top: plot x is across (0-68 m), plot y is metres from the goal line.
// The source's across coordinate runs to the shooter's right, which is already the right-hand side here.
const W = 68, HALF = 52.5;
const across = (y: number) => (y / 100) * W;
const along = (x: number) => (x / 100) * 105;

function halfPitch(): [number, number][][] {
  const arc = (cx: number, cy: number, r: number, a0: number, a1: number) =>
    Array.from({ length: 41 }, (_, i) => { const a = a0 + ((a1 - a0) * i) / 40; return [cx + r * Math.cos(a), cy + r * Math.sin(a)] as [number, number]; });
  const box = (depth: number, width: number): [number, number][] =>
    [[(W - width) / 2, 0], [(W - width) / 2, depth], [(W + width) / 2, depth], [(W + width) / 2, 0]];
  const d = Math.acos(5.5 / 9.15);
  return [
    [[0, 0], [W, 0], [W, HALF], [0, HALF], [0, 0]], box(16.5, 40.32), box(5.5, 18.32),
    arc(W / 2, 11, 9.15, Math.PI / 2 - d, Math.PI / 2 + d), arc(W / 2, HALF, 9.15, Math.PI, 2 * Math.PI),
    [[W / 2 - 3.66, 0], [W / 2 - 3.66, -1.5], [W / 2 + 3.66, -1.5], [W / 2 + 3.66, 0]],
  ];
}

function pitchPlot(width: number, marks: Plot.Markish[]) {
  const w = Math.min(width, 440);
  return Plot.plot({
    width: w, height: Math.round((w - 8) * (HALF + 3) / (W + 2)) + 8,
    style: { fontFamily: "inherit", fontSize: "12px", background: "transparent", overflow: "visible" },
    marginLeft: 4, marginRight: 4, marginTop: 4, marginBottom: 4,
    x: { domain: [-1, W + 1], axis: null }, y: { domain: [-2, HALF + 1], axis: null, reverse: true },
    r: { domain: [0, 1], range: [3, 16] },
    marks: [
      ...marks,
      ...halfPitch().map((l) => Plot.line(l, { stroke: color.muted, strokeWidth: 1.1 })),
      Plot.dot([[W / 2, 11]], { r: 1.5, fill: color.muted }),
    ],
  });
}

/** The views a position gets, in order (the first opens). A defender's shot maps only when he shoots. */
function viewsFor(position: number, shoots: boolean): View[] {
  if (position === 1) return ["faced", "keeping"];
  if (position === 2) return ["dc", ...(shoots ? (["zones", "shots"] as View[]) : [])];
  if (position === 3) return ["zones", "shots", "creating", "dc"];
  return ["zones", "shots", "creating"];
}

const HEADINGS: Record<number, string> = { 1: "In goal", 2: "Defending and going forward", 3: "Where he plays", 4: "Where he plays" };
const SIMILAR_NOTES: Record<number, string> = {
  1: "saves, the share of shots on target saved, goals prevented, the xG on target he faces and how often he's on the ball",
  2: "tackles, interceptions, clearances, blocks, recoveries and aerials won, and how far he gets forward (crosses, chances created, final-third passes, box touches)",
  3: "where he gets the ball (touches, box touches, final-third passes), what he makes of it (chances, xA, dribbles, shots and how good they are) and how much he wins back (tackles, recoveries)",
  4: "box touches, final-third passes, chances created, shots, where the shots come from and how good they are",
};

export function Profile({ player }: { player: Player }) {
  const current = useSite();
  const seasons = current.meta.seasons ?? [current.meta.season];
  const [season, setSeason] = useState(current.meta.season);
  const loaded = useSeason(season);
  const site = loaded ?? current;
  const data = useData<Profiles>(loaded ? `${loaded.root}profiles.json` : null);
  // The player as he was that season: FPL renumbers ids every season, `code` stays.
  const them = loaded ? loaded.players.find((p) => p.code === player.code) : undefined;
  const id = them?.id ?? -1;
  const position = player.element_type;
  const when = season === current.meta.season ? "this season" : `in ${season}`;
  const [picked, setView] = useState<View | null>(null);
  const all = useMemo(() => rows<Metrics>(data?.players), [data]);
  const me = all.find((m) => m.element === id);
  const myShots = useMemo(() => rows<Shot>(data?.shots).filter((s) => s.element === id), [data, id]);
  const faced = useMemo(() => rows<Faced>(data?.faced).filter((s) => s.element === id), [data, id]);
  const games = useMemo(() => rows<Game>(data?.games).filter((g) => g.element === id), [data, id]);
  const avg = data?.averages[String(position)] ?? {};
  const target = data?.dc_threshold?.[String(position)] ?? (position === 2 ? 10 : 12);

  const opponent = useCallback((fixture: number) => {
    const fx = site.fixtures.find((f) => f.id === fixture);
    return fx ? site.team.get(fx.home === them?.team ? fx.away : fx.home)?.short ?? "" : "";
  }, [site, them]);

  // ---------------------------------------------------------------- shots he took

  const zoneChart = useCallback((width: number) => {
    const zones = (data?.zones ?? []).map(([zone, x0, x1, y0, y1]) => {
      const inZone = myShots.filter((s) => s.start_x >= x0 && s.start_x < x1 && s.start_y >= y0 && s.start_y < y1);
      return { zone, x0: along(x0), x1: along(Math.min(x1, 50)), y0: across(y0), y1: across(y1), shots: inZone.length,
               xg: inZone.reduce((t, s) => t + s.xg, 0), goals: inZone.filter((s) => s.outcome === "goal").length };
    });
    const most = Math.max(1, ...zones.map((z) => z.shots));
    const total = Math.max(1, myShots.length);
    return pitchPlot(width, [
      Plot.rect(zones, { x1: "y0", x2: "y1", y1: "x0", y2: "x1", fill: color.s1, fillOpacity: (z) => (z.shots / most) * 0.85,
                         stroke: color.surface, strokeWidth: 2 }),
      Plot.text(zones.filter((z) => z.shots > 0), { x: (z) => (z.y0 + z.y1) / 2, y: (z) => (z.x0 + z.x1) / 2,
                text: (z) => pct(z.shots / total), fill: color.ink, stroke: color.surface, strokeWidth: 3, fontSize: 11 }),
      Plot.tip(zones.filter((z) => z.shots > 0), Plot.pointer({
        x: (z) => (z.y0 + z.y1) / 2, y: (z) => (z.x0 + z.x1) / 2, maxRadius: 40,
        title: (z) => `${z.shots} of his ${myShots.length} shots (${pct(z.shots / total)})\n${dec(z.xg)} xG · ${z.goals} goal${z.goals === 1 ? "" : "s"}`,
      })),
    ]);
  }, [data, myShots]);

  const shotChart = useCallback((width: number) => {
    const placed = myShots.map((s) => ({ ...s, x: across(s.start_y), y: along(s.start_x), goal: s.outcome === "goal" }))
      .sort((a, b) => b.xg - a.xg);
    const title = (s: Shot) =>
      `GW${s.gw} v ${opponent(s.fixture)} · ${s.minute}'\n${OUTCOMES[s.outcome] ?? s.outcome} · xG ${dec(s.xg)}\n${s.situation.replace(/-/g, " ")}, ${s.body_part.replace(/-/g, " ")}`;
    return pitchPlot(width, [
      Plot.dot(placed.filter((s) => !s.goal), { x: "x", y: "y", r: "xg", fill: color.s1, fillOpacity: 0.18, stroke: color.s1, strokeWidth: 1.5 }),
      Plot.dot(placed.filter((s) => s.goal), { x: "x", y: "y", r: "xg", fill: color.s1, stroke: color.surface, strokeWidth: 2 }),
      Plot.tip(placed, Plot.pointer({ x: "x", y: "y", title, maxRadius: 24 })),
    ]);
  }, [myShots, opponent]);

  // ---------------------------------------------------------------- a keeper's shots faced

  const facedChart = useCallback((width: number) => {
    const w = Math.min(width, 560);
    const placed = faced.map((s) => ({ ...s, goal: s.outcome === "goal" })).sort((a, b) => (b.xgot ?? 0) - (a.xgot ?? 0));
    const frame: [number, number][] = [[44.8, 0], [44.8, 38], [55.2, 38], [55.2, 0]];
    const fill = (s: { goal: boolean }) => (s.goal ? color.s2 : color.s1);
    const title = (s: Faced) => {
      const by = s.shooter != null ? site.player.get(s.shooter)?.web_name : undefined;
      return `GW${s.gw} v ${opponent(s.fixture)} · ${s.minute}'${by ? `\n${by}` : ""}\n${s.outcome === "goal" ? "Goal" : "Saved"} · xG on target ${dec(s.xgot ?? s.xg)}`;
    };
    return Plot.plot({
      ...plotDefaults(w),
      width: w, height: Math.round(w * 0.5), marginLeft: 8, marginRight: 8, marginTop: 12, marginBottom: 16,
      // As the keeper sees it, facing out: the shooter's right (the source's goal-mouth y rising) is his left.
      x: { domain: [57, 43], axis: null }, y: { domain: [-2, 44], axis: null },
      r: { domain: [0, 1], range: [4, 16] },
      marks: [
        Plot.ruleY([0], { stroke: color.muted }),
        Plot.line(frame, { stroke: color.ink2, strokeWidth: 3 }),
        Plot.dot(placed, { x: "goal_mouth_y", y: "goal_mouth_z", r: (s: Faced) => s.xgot ?? s.xg, fill,
                           fillOpacity: (s: { goal: boolean }) => (s.goal ? 1 : 0.18), stroke: (s: { goal: boolean }) => (s.goal ? color.surface : color.s1),
                           strokeWidth: (s: { goal: boolean }) => (s.goal ? 2 : 1.5) }),
        Plot.tip(placed, Plot.pointer({ x: "goal_mouth_y", y: "goal_mouth_z", title, maxRadius: 24 })),
      ],
    });
  }, [faced, opponent, site]);

  // ---------------------------------------------------------------- match by match

  const keepingChart = useCallback((width: number) => {
    const parts = games.flatMap((g) => [
      { gw: g.gw, part: "Saved", n: g.saves ?? 0, g }, { gw: g.gw, part: "Conceded", n: g.conceded ?? 0, g },
    ]);
    const title = (g: Game) => {
      return `GW${g.gw} v ${opponent(g.fixture)} · ${g.minutes} min\n${g.saves ?? 0} saves · ${g.conceded ?? 0} conceded` +
        (g.xgot_faced != null ? `\nxG on target faced ${dec(g.xgot_faced)} · goals prevented ${dec(g.goals_prevented)}` : "");
    };
    return Plot.plot({
      ...plotDefaults(width),
      height: 220,
      x: { label: null, type: "band", tickFormat: (d: number) => `GW${d}`, padding: barPadding(width, new Set(games.map((g) => g.gw)).size) },
      y: { label: "Shots on target", grid: true, nice: true, tickFormat: (d: number) => (Number.isInteger(d) ? `${d}` : "") },
      color: { domain: ["Saved", "Conceded"], range: [color.s1, color.s2] },
      marks: [
        Plot.ruleY([0], { stroke: color.grid }),
        Plot.barY(parts, { x: "gw", y: "n", fill: "part", insetLeft: 1, insetRight: 1 }),
        Plot.tip(games, Plot.pointerX({ x: "gw", y: (g: Game) => (g.saves ?? 0) + (g.conceded ?? 0), title })),
      ],
    });
  }, [games, opponent]);

  const dcChart = useCallback((width: number) => {
    const rest = position === 2 ? "Tackles" : "Tackles and recoveries";
    const parts = games.flatMap((g) => [
      { gw: g.gw, part: "Clearances, blocks, interceptions", n: g.cbi ?? 0, g },
      { gw: g.gw, part: rest, n: Math.max(0, (g.dc ?? 0) - (g.cbi ?? 0)), g },
    ]);
    const hits = games.filter((g) => (g.dc ?? 0) >= target);
    const title = (g: Game) => {
      return `GW${g.gw} v ${opponent(g.fixture)} · ${g.minutes} min\n${g.dc ?? 0} defensive actions (target ${target})${(g.dc ?? 0) >= target ? " · +2 points" : ""}\n` +
        `${g.cbi ?? 0} clearances, blocks, interceptions · ${Math.max(0, (g.dc ?? 0) - (g.cbi ?? 0))} ${rest.toLowerCase()}`;
    };
    return Plot.plot({
      ...plotDefaults(width),
      height: 230,
      x: { label: null, type: "band", tickFormat: (d: number) => `GW${d}`, padding: barPadding(width, new Set(games.map((g) => g.gw)).size) },
      y: { label: "Defensive actions", grid: true, nice: true, domain: [0, Math.max(target + 3, ...games.map((g) => (g.dc ?? 0) + 2))] },
      color: { domain: ["Clearances, blocks, interceptions", rest], range: [color.s1, color.s3] },
      marks: [
        Plot.ruleY([0], { stroke: color.grid }),
        Plot.barY(parts, { x: "gw", y: "n", fill: "part", insetLeft: 1, insetRight: 1 }),
        Plot.ruleY([target], { stroke: color.ink2, strokeDasharray: "4 3" }),
        Plot.text([target], { y: (d: number) => d, text: () => `target ${target}`, dy: -7, frameAnchor: "right",
                              textAnchor: "end", fill: color.ink2, fontSize: 11 }),
        Plot.text(hits, { x: "gw", y: (g: Game) => g.dc ?? 0, text: () => "+2", dy: -8, fill: color.ink, fontWeight: 600 }),
        Plot.tip(games, Plot.pointerX({ x: "gw", y: (g: Game) => g.dc ?? 0, title })),
      ],
    });
  }, [games, opponent, position, target]);

  const creatingChart = useCallback((width: number) => {
    const title = (g: Game) => `GW${g.gw} v ${opponent(g.fixture)} · ${g.minutes} min\n${g.chances_created ?? 0} chances created · xA ${dec(g.xa)}`;
    const most = Math.max(2, ...games.map((g) => g.chances_created ?? 0));
    return Plot.plot({
      ...plotDefaults(width),
      height: 220,
      marginRight: 40,
      x: { label: null, type: "band", tickFormat: (d: number) => `GW${d}`, padding: barPadding(width, new Set(games.map((g) => g.gw)).size) },
      y: { label: "Chances created", grid: true, domain: [0, most], tickFormat: (d: number) => (Number.isInteger(d) ? `${d}` : "") },
      marks: [
        Plot.ruleY([0], { stroke: color.grid }),
        Plot.barY(games, Plot.groupX({ y: "sum" }, { x: "gw", y: (g: Game) => g.chances_created ?? 0, fill: color.s1, insetLeft: 1, insetRight: 1 })),
        // xA on the same scale as chances (xA of 1 = a chance's height): the share of chances that were good ones.
        Plot.line(games, Plot.groupX({ y: "sum" }, { x: "gw", y: (g: Game) => g.xa ?? 0, stroke: color.s2, strokeWidth: 2 })),
        Plot.dot(games, Plot.groupX({ y: "sum" }, { x: "gw", y: (g: Game) => g.xa ?? 0, fill: color.s2, r: 4, stroke: color.surface, strokeWidth: 2 })),
        Plot.tip(games, Plot.pointerX({ x: "gw", y: (g: Game) => g.chances_created ?? 0, title })),
      ],
    });
  }, [games, opponent]);

  const picker = seasons.length > 1 && (
    <div className="toolbar">
      <Segmented label="Season" value={season} onChange={setSeason}
                 options={[...seasons].reverse().map((s) => ({ value: s, label: s }))} />
    </div>
  );
  const heading = <><h3>{HEADINGS[position] ?? "Where he plays"}</h3>{picker}</>;
  if (loaded === undefined || (loaded && data === undefined)) return <>{heading}<Loading /></>;
  if (!loaded || !data) return <>{heading}<p className="note">No match data for {season}.</p></>;
  if (!me) {
    return <>{heading}<p className="note">No match data for him {when}{season !== current.meta.season && !them
      ? " (he wasn't a Premier League player then)" : ""}.</p></>;
  }

  // ---------------------------------------------------------------- tiles

  const label = POSITIONS[position];
  const fewShots = me.shots < data.min_shots;
  /** A tile for a metric, against the position's average; left out when the season's data doesn't have it. */
  const tile = (title: string, key: keyof Metrics, opts: { digits?: number; percent?: boolean; per90?: boolean; unit?: string; shots?: boolean } = {}) => {
    const { digits = 1, percent = false, per90 = true, unit = "", shots = false } = opts;
    const value = me[key] as Num, average = avg[key as string];
    const f = (v: Num) => (percent ? pct(v) : `${dec(v, digits)}${unit}`);
    if (shots && fewShots) return { label: title, value: "–", note: `needs ${data.min_shots} shots` };
    if (value == null && average == null) return null;
    return { label: title, value: value == null ? "–" : `${f(value)}${per90 ? " per 90" : ""}`,
             note: average == null ? undefined : `${label} average ${f(average)}` };
  };
  const keep = <T,>(xs: (T | null)[]) => xs.filter((x): x is T => x !== null);
  const sixty = games.filter((g) => g.minutes >= 60);
  const dcTiles = keep([
    tile("Defensive contribution", "dc_p90"),
    me.dc_hits != null ? { label: `Reached ${target}`, value: `${me.dc_hits} of ${games.length} matches`, note: `+2 points each time` } : null,
  ]);
  const groups: { title?: string; tiles: { label: string; value: string; note?: string }[] }[] =
    position === 1 ? [{ tiles: keep([
      tile("Saves", "saves_p90"),
      tile("Shots on target saved", "save_share", { percent: true, per90: false }),
      me.goals_prevented != null ? { label: "Goals prevented", value: signedDec(me.goals_prevented),
        note: `xG on target faced minus goals let in: above 0, fewer than an average keeper would${avg.goals_prevented_p90 != null
          ? ` (${label} average ${signedDec(avg.goals_prevented_p90)} per 90)` : ""}` } : null,
      tile("xG on target faced", "xgot_faced_p90", { digits: 2 }),
      tile("Goals conceded", "conceded_p90", { digits: 2 }),
      { label: "Clean sheets", value: `${sixty.filter((g) => (g.conceded ?? 0) === 0).length} of ${sixty.length}`, note: "matches he played 60+ minutes in" },
    ]) }]
    : position === 2 ? [
      { title: "Defending", tiles: [...dcTiles, ...keep([
        tile("Clearances", "clearances_p90"), tile("Interceptions", "interceptions_p90"), tile("Blocks", "blocks_p90"),
        tile("Tackles", "tackles_p90"), tile("Aerial duels won", "aerials_won_p90")])] },
      { title: "Going forward", tiles: keep([
        tile("Chances created", "chances_created_p90", { digits: 2 }), tile("Crosses completed", "crosses_p90", { digits: 2 }),
        tile("Passes into the final third", "final_third_passes_p90"), tile("Touches in the opposition box", "box_touches_p90"),
        tile("Shots", "shots_p90", { digits: 2 })]) },
    ]
    : position === 3 ? [
      { title: "Attacking", tiles: keep([
        tile("Touches in the opposition box", "box_touches_p90"), tile("Chances created", "chances_created_p90", { digits: 2 }),
        tile("xA", "xa_p90", { digits: 2 }), tile("Passes into the final third", "final_third_passes_p90"),
        tile("Dribbles completed", "dribbles_p90"), tile("Shots", "shots_p90"),
        tile("xG per shot", "xg_per_shot", { digits: 2, per90: false, shots: true })]) },
      { title: "Defending", tiles: [...dcTiles, ...keep([tile("Recoveries", "recoveries_p90"), tile("Tackles", "tackles_p90")])] },
    ]
    : [{ tiles: keep([
      tile("Touches in the opposition box", "box_touches_p90"), tile("Passes into the final third", "final_third_passes_p90"),
      tile("Shots", "shots_p90"), tile("Shots from inside the box", "box_shot_share", { percent: true, per90: false, shots: true }),
      tile("Average shot distance", "shot_distance", { per90: false, unit: " m", shots: true }),
      tile("xG per shot", "xg_per_shot", { digits: 2, per90: false, shots: true }),
      tile("Chances created", "chances_created_p90", { digits: 2 })]) }];

  // ---------------------------------------------------------------- views

  const views = viewsFor(position, myShots.length >= data.min_shots).filter((v) =>
    (v === "zones" || v === "shots") ? myShots.length > 0 : v === "faced" ? faced.length > 0 : games.length > 0);
  const view = picked && views.includes(picked) ? picked : views[0];
  const charts: Record<View, { make: (w: number) => ReturnType<typeof Plot.plot>; height: number; aria: string; pitch: boolean;
                               legend?: { label: string; color: string; kind?: "dot" | "line" }[]; note: string }> = {
    zones: { make: zoneChart, height: 360, pitch: true, aria: "His shots by zone round the goal",
             note: `His ${myShots.length} shots ${when} by where they were taken: the darker the zone, the more of them came from there. Hover a zone for its xG and goals.` },
    shots: { make: shotChart, height: 360, pitch: true, aria: `Every shot he took ${when}`,
             note: "Every shot, where it was taken: the bigger the circle, the better the chance (xG); solid circles are goals. Hover for the match and minute." },
    faced: { make: facedChart, height: 260, pitch: false, aria: `The shots on target he faced ${when}, on the goal frame`,
             legend: [{ label: "Saved", color: color.s1, kind: "dot" }, { label: "Goal", color: color.s2, kind: "dot" }],
             note: `The ${faced.length} shots on target he faced ${when}, where they crossed the line, as he sees them from his goal line. The bigger the circle, the more likely a shot placed there was to go in (xG on target); goals are solid. Hover for the shooter and the match.` },
    keeping: { make: keepingChart, height: 220, pitch: false, aria: "Saves and goals conceded match by match",
               legend: [{ label: "Saved", color: color.s1 }, { label: "Conceded", color: color.s2 }],
               note: "Each match's shots on target: saved and let in. FPL pays a point for every 3 saves; goalkeepers lose one for every 2 conceded (60+ minutes). Hover for the xG on target he faced." },
    dc: { make: dcChart, height: 230, pitch: false, aria: "Defensive contribution match by match against FPL's target",
          legend: [{ label: "Clearances, blocks, interceptions", color: color.s1 }, { label: position === 2 ? "Tackles" : "Tackles and recoveries", color: color.s3 }],
          note: `FPL's own count of his defensive actions each match. Reaching ${target} (the dashed line) is worth 2 points: ${position === 2
            ? "for defenders, clearances, blocks, interceptions and tackles count" : "for midfielders and forwards, recoveries count too, and the target is 12"}.` },
    creating: { make: creatingChart, height: 220, pitch: false, aria: "Chances created and xA match by match",
                legend: [{ label: "Chances created", color: color.s1 }, { label: "xA", color: color.s2, kind: "line" }],
                note: "The chances he set up each match, and their xA (the chance each became a goal, added up): many chances with little xA are long shots for teammates. FPL pays 3 points an assist." },
  };
  const shown = view ? charts[view] : null;

  // Similar players are that season's; linked (with price and xP) where they're still in the game.
  const now = new Map(current.players.map((p) => [p.code, p]));
  const similar = (data.similar[String(id)] ?? []).map(([pid, sim]) => ({ p: site.player.get(pid), sim }))
    .filter((x) => x.p).map(({ p, sim }) => ({ p: p!, sim, today: now.get(p!.code) }));

  return (
    <>
      {heading}
      {season !== current.meta.season && them && them.team !== undefined && (
        <p className="note" style={{ marginTop: 0 }}>At <Club code={site.team.get(them.team)?.code} /> by the end of {season}; the numbers cover all his matches that season.</p>
      )}
      {groups.map((g, i) => (
        <div key={g.title ?? i}>
          {g.title && <h4 className="profile-sub">{g.title}</h4>}
          <Tiles tiles={g.tiles} />
        </div>
      ))}
      {position === 2 && myShots.length > 0 && myShots.length < data.min_shots && (
        <p className="note">{myShots.length} shot{myShots.length === 1 ? "" : "s"} {when}: shot maps show for a defender with {data.min_shots} or more.</p>
      )}
      {shown && (
        <div className={shown.pitch ? "profile-pitch" : "profile-chart"}>
          {views.length > 1 && (
            <Segmented label="View" value={view} onChange={setView} options={views.map((v) => ({ value: v, label: VIEW_LABELS[v] }))} />
          )}
          {shown.legend && <Legend items={shown.legend} />}
          <Chart make={shown.make} height={shown.height} ariaLabel={shown.aria} />
          <p className="note" style={{ marginTop: 0 }}>{shown.note}</p>
        </div>
      )}
      {similar.length > 0 && (
        <>
          <h3>Similar profiles{season !== current.meta.season && ` in ${season}`}</h3>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Player</th><th>Club</th><th className="num">£m</th><th className="num">Similarity</th>
                <th className="num">{current.meta.next_gw ? `xP GW${current.meta.next_gw}` : "xP next"}</th></tr></thead>
              <tbody>
                {similar.map(({ p, sim, today }) => (
                  <tr key={p.id} className={today ? "clickable" : undefined}
                      onClick={today ? () => { window.location.hash = `players/${today.id}`; } : undefined}>
                    <td>{p.web_name}</td><td><Club code={site.team.get(p.team)?.code} /></td>
                    <td className="num">{today ? money(today.now_cost) : "–"}</td>
                    <td className="num">{pct(sim)}</td><td className="num">{today ? pts(today.forecast) : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Note>{label}s with the most similar numbers: {SIMILAR_NOTES[position] ?? SIMILAR_NOTES[4]}, each compared with the
            position's average (cosine similarity). Only players with {data.min_minutes}+ minutes are compared. It says who plays a
            similar role, not who will score more: that's the xP.
            {season !== current.meta.season && " The club is theirs at the end of that season; price and xP are today's, for players still in the game."}</Note>
        </>
      )}
      {similar.length === 0 && me.minutes < data.min_minutes && (
        <p className="note">Similar profiles need {data.min_minutes} minutes {when}; he has {Math.round(me.minutes)}.</p>
      )}
    </>
  );
}

function signedDec(v: number): string {
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${dec(Math.abs(v), 2)}`;
}
