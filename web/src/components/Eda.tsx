import * as Plot from "@observablehq/plot";
import { useCallback, useState } from "react";
import { color } from "../colors";
import type { Row } from "../data";
import { dec, int, pct, pts } from "../format";
import { Chart, Legend, Note, Segmented, Table, plotDefaults, type Column } from "./ui";

/** models/eda.json (eda.py): what goes with points, by position, from pre-match features. */
export interface EdaReport {
  seasons: string[];
  rows: number;
  correlations: Row[];
  spread: Row[];
  shap: Row[];
}

const POSITIONS = ["GKP", "DEF", "MID", "FWD"] as const;
type Position = (typeof POSITIONS)[number];
const MORE = "More of it, more points";
const FEWER = "More of it, fewer points";

export default function WhatGoesWithPoints({ report }: { report: EdaReport }) {
  const [pos, setPos] = useState<Position>("MID");
  const spread = report.spread.find((r) => r.position === pos);
  const corr = report.correlations.filter((r) => r.position === pos);
  const shap = report.shap.filter((r) => r.position === pos);
  return (
    <>
      <h3>What goes with points?</h3>
      <p className="note" style={{ marginTop: 0 }}>
        Everything here is as it stood <em>before</em> each match, set against that match's points: what this model sees.
        (Setting a week's stats against the same week's points would only show that goals score points.)
        Seasons {report.seasons.join(", ")}, players getting minutes ({int(report.rows)} player-matches).
      </p>
      <Segmented label="Position" value={pos} onChange={setPos} options={POSITIONS.map((p) => ({ value: p, label: p }))} />
      {spread && <Raincloud row={spread} />}
      <div className="grid-2">
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Linked with points</h3>
          {corr.length > 0 && <Bars rows={corr} value="rho" label="Rank correlation with points" sign={(r) => r.rho} />}
          <p className="note">Spearman rank correlation of each feature with the match's points. Averages of the same stat over
            different windows move together, so the list repeats itself.</p>
        </div>
        <div className="card">
          <h3 style={{ marginTop: 0 }}>What moves this model's forecast</h3>
          {shap.length > 0
            ? <Bars rows={shap} value="mean_abs" label="Average effect on xP (points)" sign={(r) => r.direction} />
            : <p className="note">Needs the LightGBM model.</p>}
          <p className="note">SHAP for the LightGBM part of this model: how far each feature moves a player's xP from the average,
            on average. A feature that only repeats another gets little here.</p>
        </div>
      </div>
      <Legend items={[{ label: MORE, color: color.s1 }, { label: FEWER, color: color.s2 }]} />
    </>
  );
}

/** One position's points per appearance: the share at each score, the box and a sample of single scores. */
function Raincloud({ row }: { row: Row }) {
  const make = useCallback((width: number) => {
    const x = { domain: [-3.5, 20.5] as [number, number] };
    const cloud = Plot.plot({
      ...plotDefaults(width), height: 140, marginBottom: 8,
      x: { ...x, axis: null },
      y: { label: "Share", tickFormat: "%", grid: true },
      marks: [
        Plot.rectY(row.histogram, { x1: (d: Row) => d.points - 0.4, x2: (d: Row) => d.points + 0.4, y: "share", fill: color.s1 }),
        Plot.tip(row.histogram, Plot.pointerX({ x: "points", y: "share",
          title: (d: Row) => `${d.points}${d.points >= 20 ? "+" : ""} points: ${pct(d.share, 1)} of appearances` })),
      ],
    });
    const jitter = row.sample.map((p: number, i: number) => ({ points: p + (((i * 37) % 11) - 5) / 30, y: ((i * 53) % 97) / 97 }));
    const rain = Plot.plot({
      ...plotDefaults(width), height: 90,
      x: { ...x, label: "Points in a match (20+ in the last bar)" },
      y: { axis: null, domain: [-0.9, 1] },
      marks: [
        Plot.dot(jitter, { x: "points", y: "y", r: 1.6, fill: color.s1, fillOpacity: 0.35 }),
        Plot.ruleY([-0.45], { x1: row.p5, x2: row.p95, stroke: color.muted, strokeWidth: 2 }),
        Plot.rect([row], { x1: "p25", x2: "p75", y1: -0.7, y2: -0.2, fill: color.s1, fillOpacity: 0.35 }),
        Plot.ruleX([row.p50], { y1: -0.75, y2: -0.15, stroke: color.s1, strokeWidth: 3 }),
      ],
    });
    const box = document.createElement("div");
    box.append(cloud, rain);
    return box;
  }, [row]);
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>How {row.position} points are spread</h3>
      <p className="note" style={{ marginTop: 0 }}>
        Per appearance: average {pts(row.mean)}, half of all scores between {dec(row.p25, 0)} and {dec(row.p75, 0)};
        {" "}{pct(row.share_2_or_fewer)} were 2 or fewer and {pct(row.share_10_plus, 1)} were 10+.
      </p>
      <Chart make={make} height={230} ariaLabel={`Spread of ${row.position} points per appearance`} />
      <p className="note">Bars: the share of appearances with each score. Below: the middle 50% (box), 5th–95th percentile (line),
        median (tick) and a sample of single scores. Most appearances are a 1 or a 2; the rare hauls pull the average up,
        which is why xP sits above the typical score.</p>
    </div>
  );
}

function Bars({ rows, value, label, sign }: { rows: Row[]; value: string; label: string; sign: (r: Row) => number }) {
  const data = [...rows].sort((a, b) => Math.abs(b[value]) - Math.abs(a[value]))
    .map((r): Row => ({ ...r, way: sign(r) >= 0 ? MORE : FEWER }));
  const make = useCallback((width: number) => Plot.plot({
    ...plotDefaults(width), height: 26 * data.length + 50, marginLeft: Math.min(210, width * 0.5),
    x: { label, grid: true },
    y: { label: null, domain: data.map((d) => d.label) },
    color: { domain: [MORE, FEWER], range: [color.s1, color.s2] },
    marks: [
      Plot.barX(data, { x: value, y: "label", fill: "way" }),
      Plot.ruleX([0], { stroke: color.muted }),
      Plot.tip(data, Plot.pointerY({ x: value, y: "label", title: (d: Row) => `${d.label}\n${label}: ${dec(d[value], 3)}\n(${d.feature})` })),
    ],
  }), [data, value, label]);
  return <Chart make={make} height={26 * data.length + 50} ariaLabel={label} />;
}

/** Log score and RPS of each forecast of the chance of every score (distribution.py). */
export function DistributionScores({ rows, season }: { rows: Row[]; season: string }) {
  const names: Record<string, string> = { dist: "Distribution model (PyTorch)" };
  const columns: Column<Row>[] = [
    { key: "model", label: "Forecast", value: (r) => names[r.model] ?? r.model },
    { key: "log_score", label: "Log score", numeric: true, value: (r) => r.log_score, render: (r) => dec(r.log_score, 3),
      title: "Minus the log of the chance given to the score that happened (lower is better)" },
    { key: "rps", label: "RPS", numeric: true, value: (r) => r.rps, render: (r) => dec(r.rps, 3),
      title: "Ranked probability score: rewards forecasts close to the score (lower is better)" },
  ];
  return (
    <>
      <h3>Is the whole spread right?</h3>
      <p className="note" style={{ marginTop: 0 }}>
        Two scores for a forecast of the chance of <em>every</em> score, lower is better: the <strong>log score</strong> (how
        surprised the forecast was by what happened, so a confident miss costs a lot) and the <strong>RPS</strong> (how far the
        forecast's spread sat from the score: 7 when 8 happened beats 2). The benchmark is what a forecaster with no model would
        say: the spread of scores past players of the same position and similar 5-match form went on to get. The distribution
        model is a PyTorch network that forecasts the chances directly instead of simulating matches.
      </p>
      <Table columns={columns} data={rows} sort="log_score" desc={false} rowKey={(r) => r.model} />
      <Note>{season}, players getting minutes ({int(rows[0]?.rows ?? 0)} player-matches).</Note>
    </>
  );
}
