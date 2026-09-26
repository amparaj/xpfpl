// The Monte Carlo (src/xpfpl/simulate.py) on the site: a team's simulated score, the captain options'
// odds, the chips' chances, and the check that the simulated chances came true.

import * as Plot from "@observablehq/plot";
import { useCallback } from "react";
import { color } from "../colors";
import type { ChipOdds, Row, Simulation, Spread } from "../data";
import { pct, pts } from "../format";
import { useSite } from "../site";
import { Chart, Legend, Table, plotDefaults, type Column } from "./ui";

export const PAPER = "https://arxiv.org/abs/2505.02170";

/** "33–69": the middle 80% of simulated scores. */
export const band = (lo: number | null | undefined, hi: number | null | undefined) =>
  lo == null || hi == null ? "–" : `${Math.round(lo)}–${Math.round(hi)}`;

/** Histogram of a team's simulated gameweek scores, with the forecast (and, once played, the score) marked. */
export function ScoreChart({ spread, forecast, actual, gw }: { spread: Spread; forecast?: number | null; actual?: number | null; gw: number }) {
  const h = spread.histogram;
  const bins = h ? h.shares.map((share, i) => ({ x1: h.start + i * h.width, x2: h.start + (i + 1) * h.width, share })) : [];
  const make = useCallback((width: number) => Plot.plot({
    ...plotDefaults(width),
    height: 200,
    x: { label: `Simulated GW${gw} points`, grid: false },
    y: { label: "Share of simulations", grid: true, tickFormat: "%" },
    marks: [
      Plot.rectY(bins, { x1: "x1", x2: "x2", y: "share", fill: color.s1, fillOpacity: 0.8, inset: 1 }),
      Plot.ruleY([0], { stroke: color.grid }),
      ...(forecast != null ? [Plot.ruleX([forecast], { stroke: color.s3, strokeWidth: 3 })] : []),
      ...(actual != null ? [Plot.ruleX([actual], { stroke: color.s2, strokeWidth: 3 })] : []),
      Plot.tip(bins, Plot.pointerX({ x: (b: Row) => (b.x1 + b.x2) / 2, y: "share",
        title: (b: Row) => `${b.x1}–${b.x2 - 1} points: ${pct(b.share, 1)} of simulations` })),
    ],
  }), [bins, forecast, actual, gw]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!h) return null;
  return (
    <>
      <Legend items={[{ label: "Simulated scores", color: color.s1 },
                      ...(forecast != null ? [{ label: "Forecast (xP)", color: color.s3, kind: "line" as const }] : []),
                      ...(actual != null ? [{ label: "Scored", color: color.s2, kind: "line" as const }] : [])]} />
      <Chart make={make} height={200} ariaLabel={`Distribution of the team's simulated GW${gw} score`} />
    </>
  );
}

type Captain = Simulation["captains"][number];

/** The captain options' odds, from the decision's simulation. */
export function CaptainOddsTable({ sim, captain }: { sim: Simulation; captain: number }) {
  const site = useSite();
  const columns: Column<Captain>[] = [
    { key: "player", label: "Player", value: (c) => site.player.get(c.element)?.web_name ?? c.element,
      render: (c) => <>{site.player.get(c.element)?.web_name ?? c.element}{c.element === captain ? <span className="tag" style={{ marginLeft: 6 }}>C</span> : null}</> },
    { key: "mean", label: "Average", numeric: true, value: (c) => c.mean, render: (c) => pts(c.mean),
      title: "His average points in the simulations: his xP, give or take the simulation's rounding" },
    { key: "range", label: "Middle 80%", numeric: true, value: (c) => c.pts_p90, render: (c) => band(c.pts_p10, c.pts_p90),
      title: "His points in the middle 80% of simulated weeks, before the armband" },
    { key: "haul", label: "10+", numeric: true, value: (c) => c.p_haul, render: (c) => pct(c.p_haul) },
    { key: "blank", label: "2 or fewer", numeric: true, value: (c) => c.p_blank, render: (c) => pct(c.p_blank) },
    { key: "best", label: "Best pick", numeric: true, value: (c) => c.p_best, render: (c) => <strong>{pct(c.p_best)}</strong>,
      title: "How often he outscores every other option in the same simulated week" },
  ];
  return <Table columns={columns} data={sim.captains} sort="mean" rowKey={(c) => c.element} />;
}

const CHIP_LABELS: Record<string, string> = { "3xc": "Triple Captain", bboost: "Bench Boost" };
const CHIP_GAIN: Record<string, string> = { "3xc": "the captain's points once more", bboost: "the bench's points" };

/** One line per chip: what it would add in simulation and how often that clears the threshold. */
export function ChipOddsList({ chips }: { chips: Record<string, ChipOdds> }) {
  const entries = Object.entries(chips);
  if (!entries.length) return null;
  return (
    <ul style={{ margin: "6px 0 0" }}>
      {entries.map(([chip, o]) => (
        <li key={chip}>
          <strong>{CHIP_LABELS[chip] ?? chip}</strong>: would add {band(o.p10, o.p90)} points ({CHIP_GAIN[chip]}, middle 80%);
          reaches the {o.threshold}-point threshold in {pct(o.p_clear)} of simulated weeks
          {o.p_beats_later != null && <>, and beats playing it in a later week of the plan in {pct(o.p_beats_later)}</>}.
        </li>
      ))}
    </ul>
  );
}

/** Simulated chance against how often it happened, for the players grouped by that chance. */
export function Reliability({ rows, label }: { rows: Row[]; label: string }) {
  const max = Math.min(1, Math.max(...rows.map((r) => Math.max(r.predicted, r.actual + 2 * r.se))) * 1.08);
  const make = useCallback((width: number) => Plot.plot({
    ...plotDefaults(width),
    height: 240,
    x: { label: `Simulated chance of ${label}`, domain: [0, max], grid: true, tickFormat: "%" },
    y: { label: "How often it happened", domain: [0, max], grid: true, tickFormat: "%" },
    marks: [
      Plot.line([[0, 0], [max, max]], { stroke: color.muted, strokeDasharray: "4,4", strokeWidth: 1 }),
      Plot.ruleX(rows, { x: "predicted", y1: (r: Row) => Math.max(0, r.actual - 2 * r.se), y2: (r: Row) => r.actual + 2 * r.se, stroke: color.s1, strokeWidth: 2 }),
      Plot.dot(rows, { x: "predicted", y: "actual", r: 5, fill: color.s1, stroke: color.surface, strokeWidth: 2 }),
      Plot.tip(rows, Plot.pointer({ x: "predicted", y: "actual",
        title: (r: Row) => `Given ${r.bin}: simulated ${pct(r.predicted, 1)}, happened ${pct(r.actual, 1)}\n${r.n.toLocaleString()} player-matches` })),
    ],
  }), [rows, max, label]);
  return <Chart make={make} height={240} ariaLabel={`Simulated chance of ${label} against how often it happened`} />;
}
