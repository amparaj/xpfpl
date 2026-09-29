// Where a player plays (profiles.json, spatial.py): his shots binned into zones round the goal and
// every shot on a half pitch, per-90 zone metrics against his position's average, and the players
// whose profile is most like his. There are no heatmaps in the source: shot locations and zone
// counts (touches in the box, final-third passes) are the spatial data there is.

import * as Plot from "@observablehq/plot";
import { useCallback, useMemo, useState } from "react";
import { color } from "../colors";
import type { Player } from "../data";
import { rows } from "../data";
import { POSITIONS, dec, money, pct, pts } from "../format";
import { useData, useSite } from "../site";
import { Chart, Club, Note, Segmented, Tiles } from "./ui";

interface Profiles {
  zones: [number, number, number, number, number][];      // zone, x0, x1 (from the goal line), y0, y1 (across)
  min_minutes: number; min_shots: number;
  players: Record<string, unknown[]>;
  averages: Record<string, Record<string, number | null>>;
  similar: Record<string, [number, number][]>;
  shots: Record<string, unknown[]>;
}
interface Metrics {
  element: number; minutes: number; matches: number; shots: number; xg: number; goals: number;
  touches_p90: number; box_touches_p90: number; final_third_passes_p90: number; chances_created_p90: number;
  shots_p90: number; box_shot_share: number | null; shot_distance: number | null; xg_per_shot: number | null;
  header_share: number | null;
}
interface Shot {
  element: number; gw: number; fixture: number; minute: number; start_x: number; start_y: number; xg: number;
  outcome: string; situation: string; body_part: string;
}

const OUTCOMES: Record<string, string> = {
  goal: "Goal", save: "Saved", miss: "Off target", block: "Blocked", post: "Hit the woodwork", "blocked-off-line": "Cleared off the line",
};
// Half a 105 x 68 pitch, drawn with the goal at the top: plot x is across (0-68 m), plot y is metres from the goal line.
// The source's across coordinate runs to the shooter's left, so it's mirrored to put his left on the left.
const W = 68, HALF = 52.5;
const across = (y: number) => W - (y / 100) * W;
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

export function Profile({ player }: { player: Player }) {
  const site = useSite();
  const data = useData<Profiles>("profiles.json");
  const [view, setView] = useState<"zones" | "shots">("zones");
  const all = useMemo(() => rows<Metrics>(data?.players), [data]);
  const me = all.find((m) => m.element === player.id);
  const myShots = useMemo(() => rows<Shot>(data?.shots).filter((s) => s.element === player.id), [data, player.id]);
  const avg = data?.averages[String(player.element_type)] ?? {};

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
    const title = (s: Shot & { x: number; y: number }) => {
      const fx = site.fixtures.find((f) => f.id === s.fixture);
      const opp = fx ? site.team.get(fx.home === player.team ? fx.away : fx.home)?.short : "";
      return `GW${s.gw} v ${opp} · ${s.minute}'\n${OUTCOMES[s.outcome] ?? s.outcome} · xG ${dec(s.xg)}\n${s.situation.replace(/-/g, " ")}, ${s.body_part.replace(/-/g, " ")}`;
    };
    return pitchPlot(width, [
      Plot.dot(placed.filter((s) => !s.goal), { x: "x", y: "y", r: "xg", fill: color.s1, fillOpacity: 0.18, stroke: color.s1, strokeWidth: 1.5 }),
      Plot.dot(placed.filter((s) => s.goal), { x: "x", y: "y", r: "xg", fill: color.s1, stroke: color.surface, strokeWidth: 2 }),
      Plot.tip(placed, Plot.pointer({ x: "x", y: "y", title, maxRadius: 24 })),
    ]);
  }, [myShots, site, player.team]);

  if (data === undefined || data === null) return null;
  if (!me) return <p className="note">No match data for him yet this season.</p>;
  const vs = (key: keyof Metrics, digits = 2, asPct = false) => {
    const a = avg[key as string];
    const f = (v: number | null | undefined) => (asPct ? pct(v) : dec(v, digits));
    return a === null || a === undefined ? undefined : `${POSITIONS[player.element_type]} average ${f(a)}`;
  };
  const fewShots = me.shots < data.min_shots;
  const similar = (data.similar[String(player.id)] ?? []).map(([id, sim]) => ({ p: site.player.get(id), sim })).filter((x) => x.p);

  return (
    <>
      <h3>Where he plays</h3>
      <Tiles tiles={[
        { label: "Touches in the opposition box", value: `${dec(me.box_touches_p90, 1)} per 90`, note: vs("box_touches_p90", 1) },
        { label: "Passes into the final third", value: `${dec(me.final_third_passes_p90, 1)} per 90`, note: vs("final_third_passes_p90", 1) },
        { label: "Shots", value: `${dec(me.shots_p90, 1)} per 90`, note: vs("shots_p90", 1) },
        { label: "Shots from inside the box", value: fewShots ? "–" : pct(me.box_shot_share), note: fewShots ? `needs ${data.min_shots} shots` : vs("box_shot_share", 0, true) },
        { label: "Average shot distance", value: fewShots ? "–" : `${dec(me.shot_distance, 1)} m`, note: fewShots ? undefined : vs("shot_distance", 1) },
        { label: "xG per shot", value: fewShots ? "–" : dec(me.xg_per_shot), note: fewShots ? undefined : vs("xg_per_shot") },
      ]} />
      {myShots.length > 0 && (
        <div className="profile-pitch">
          <Segmented label="Shot view" value={view} onChange={setView}
                     options={[{ value: "zones", label: "Shot zones" }, { value: "shots", label: "Every shot" }]} />
          <Chart make={view === "zones" ? zoneChart : shotChart} height={360}
                 ariaLabel={view === "zones" ? "His shots by zone round the goal" : "Every shot he has taken this season"} />
          <p className="note" style={{ marginTop: 0 }}>
            {view === "zones"
              ? `His ${myShots.length} shots this season by where they were taken: the darker the zone, the more of them came from there. Hover a zone for its xG and goals.`
              : "Every shot, where it was taken: the bigger the circle, the better the chance (xG); solid circles are goals. Hover for the match and minute."}
          </p>
        </div>
      )}
      {similar.length > 0 && (
        <>
          <h3>Similar profiles</h3>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Player</th><th>Club</th><th className="num">£m</th><th className="num">Similarity</th>
                <th className="num">{site.meta.next_gw ? `xP GW${site.meta.next_gw}` : "xP next"}</th></tr></thead>
              <tbody>
                {similar.map(({ p, sim }) => (
                  <tr key={p!.id} className="clickable" onClick={() => { window.location.hash = `players/${p!.id}`; }}>
                    <td>{p!.web_name}</td><td><Club id={p!.team} /></td><td className="num">{money(p!.now_cost)}</td>
                    <td className="num">{pct(sim)}</td><td className="num">{pts(p!.forecast)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Note>Players at the same position who get the ball and their shots in the most similar places: box touches, final-third
            passes, chances created, shots, where the shots come from and how good they are, each compared with the position's
            average (cosine similarity). Only players with {data.min_minutes}+ minutes are compared. It says who plays a similar
            role, not who will score more: that's the xP.</Note>
        </>
      )}
      {similar.length === 0 && me.minutes < data.min_minutes && (
        <p className="note">Similar profiles need {data.min_minutes} minutes this season; he has {Math.round(me.minutes)}.</p>
      )}
    </>
  );
}
