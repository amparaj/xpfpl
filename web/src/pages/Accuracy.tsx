import * as Plot from "@observablehq/plot";
import { useCallback, useState } from "react";
import { color } from "../colors";
import { Chart, Legend, Loading, Note, Table, plotDefaults, type Column } from "../components/ui";
import type { Accuracy as AccuracyFile, Row } from "../data";
import { dec, pts, signed } from "../format";
import { useData, useSite } from "../site";

export default function Accuracy() {
  const site = useSite();
  const data = useData<AccuracyFile>("accuracy.json");
  if (data === undefined) return <Loading />;
  if (!data) return <p>No accuracy reports exported yet.</p>;
  const { validation: v, comparison: c, scorecard: s, tuning: t } = data;

  return (
    <>
      <h2>How accurate is the model?</h2>
      <p className="lede">
        The model predicts each player's points (xP) before every deadline. Here it is scored on a whole season it never
        saw in training{v ? ` (${v.season})` : ""}, and on this season's forecasts as the results come in.
      </p>

      <h3>This season, forecast by forecast</h3>
      <LiveRecord scorecard={s} />

      {c && <Comparison comparison={c} />}
      {v && <ByGameweek validation={v} comparison={c} preferred={site.meta.model} />}
      {v && <Calibration validation={v} />}
      {t && <Tuning tuning={t} />}
      <Note>
        The default model is <strong>{site.meta.model}</strong> ({site.meta.model_description}). The ceiling is what a model
        that knew every player's true chances would score, simulated: some of each week's points are luck no model can predict.
      </Note>
    </>
  );
}

function LiveRecord({ scorecard }: { scorecard: any }) {
  const site = useSite();
  const gws: Row[] = scorecard?.gameweeks ?? [];
  if (!gws.length) {
    const next = site.meta.next_gw;
    return (
      <p className="note" style={{ marginTop: 0 }}>
        Nothing to score yet. The model has been trained, but this table only counts forecasts that were saved{" "}
        <em>before</em> a deadline and then checked against the real points, so nothing can be adjusted after the
        fact. Saving started partway through the season, so the earlier gameweeks aren't here
        {next ? <>; the GW{next} forecast is saved and will be scored once GW{next} has been played</> : null}. Until
        then, the full-season test below is the best guide.
      </p>
    );
  }
  const columns: Column<Row>[] = [
    { key: "gw", label: "GW", numeric: true, value: (r) => r.gw },
    { key: "model", label: "Model", value: (r) => r.model },
    { key: "n", label: "Players", numeric: true, value: (r) => r.n },
    { key: "mae", label: "MAE", numeric: true, value: (r) => r.mae, render: (r) => pts(r.mae), title: "Average miss, in points" },
    { key: "rmse", label: "RMSE", numeric: true, value: (r) => r.rmse, render: (r) => pts(r.rmse) },
    { key: "spearman", label: "Rank corr.", numeric: true, value: (r) => r.spearman, render: (r) => dec(r.spearman), title: "How well it ordered the players (1 = perfectly)" },
    { key: "bias", label: "Bias", numeric: true, value: (r) => r.bias, render: (r) => signed(r.bias), title: "Average of xP minus points: positive means too high" },
  ];
  return <Table columns={columns} data={gws} sort="gw" rowKey={(r) => `${r.gw}-${r.model}`} />;
}

function Comparison({ comparison }: { comparison: any }) {
  const models: Row[] = comparison.models ?? [];
  const ceiling = comparison.ceiling;
  const columns: Column<Row>[] = [
    { key: "model", label: "Model", value: (r) => r.model },
    { key: "description", label: "What it is", value: (r) => r.description, wrap: true },
    { key: "rmse", label: "RMSE", numeric: true, value: (r) => r.rmse, render: (r) => dec(r.rmse, 3), title: "Root mean squared error, points (lower is better)" },
    { key: "mae", label: "MAE", numeric: true, value: (r) => r.mae, render: (r) => dec(r.mae, 3) },
    { key: "r2", label: "R²", numeric: true, value: (r) => r.r2, render: (r) => dec(r.r2, 3) },
    { key: "spearman", label: "Rank corr.", numeric: true, value: (r) => r.spearman, render: (r) => dec(r.spearman, 3) },
    { key: "captain", label: "Captain pts/GW", numeric: true, value: (r) => r.captain_pts_per_gw, render: (r) => dec(r.captain_pts_per_gw, 1),
      title: "Points per gameweek from captaining the model's top pick (noisy: one pick a week)" },
  ];
  return (
    <>
      <h3>Every model on {comparison.season}</h3>
      <Table columns={columns} data={models} sort="rmse" desc={false} rowKey={(r) => r.model} />
      <Note>
        {comparison.rows?.toLocaleString()} player-matches where the player got minutes.
        {ceiling && <> Perfect-model ceiling: RMSE {dec(ceiling.rmse_median, 2)} ({dec(ceiling.rmse_p5, 2)}–{dec(ceiling.rmse_p95, 2)}), R² {dec(ceiling.r2_median, 2)}.</>}
      </Note>
    </>
  );
}

function ByGameweek({ validation, comparison, preferred }: { validation: any; comparison: any; preferred: string }) {
  // `xpfpl compare` scores every model week by week; older exports only have the default model's
  // validation run, so fall back to that.
  const all = comparison?.by_gameweek?.length && comparison.season === validation.season;
  const rows: Row[] = all ? comparison.by_gameweek : validation.by_gameweek ?? [];
  const reference: string = all ? "baseline" : validation.reference;
  const choices: string[] = [...new Set(rows.map((r) => r.model as string))].filter((m) => m !== reference);
  const [picked, setPicked] = useState<string>(() =>
    choices.includes(preferred) ? preferred : choices.includes(validation.primary) ? validation.primary : choices[0]);
  const model = choices.includes(picked) ? picked : choices[0];
  const data = rows.filter((r) => r.model === model || r.model === reference);
  const models = [model, reference];
  const make = useCallback((width: number) => Plot.plot({
    ...plotDefaults(width),
    height: 240,
    x: { label: "Gameweek", grid: false },
    y: { label: "RMSE (points)", grid: true, zero: true },
    color: { domain: models, range: [color.s1, color.neutral] },
    marks: [
      Plot.line(data, { x: "gw", y: "rmse", stroke: "model", strokeWidth: 2 }),
      Plot.ruleX(data, Plot.pointerX({ x: "gw", stroke: color.muted })),
      Plot.tip(data, Plot.pointerX({ x: "gw", y: "rmse", title: (r: Row) => `GW${r.gw} · ${r.model}
RMSE ${pts(r.rmse)} · MAE ${pts(r.mae)}` })),
    ],
  }), [data]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Error week by week, {validation.season}</h3>
      {choices.length > 1 && (
        <div className="toolbar">
          <label>
            Model
            <select value={model} onChange={(e) => setPicked(e.target.value)}>
              {choices.map((m) => <option key={m} value={m}>{m === preferred ? `${m} (default)` : m}</option>)}
            </select>
          </label>
          <span className="muted" style={{ fontSize: 13 }}>against the baseline</span>
        </div>
      )}
      <Legend items={[{ label: model, color: color.s1, kind: "line" }, { label: `${reference} (5-match average)`, color: color.neutral, kind: "line" }]} />
      <Chart make={make} height={240} ariaLabel={`RMSE by gameweek for ${model} and the baseline`} />
      <p className="note">Lower is better. The baseline just averages each player's last five matches.</p>
    </div>
  );
}

function Calibration({ validation }: { validation: any }) {
  const data: Row[] = validation.calibration ?? [];
  const max = Math.max(1, ...data.map((d) => Math.max(d.predicted, d.actual + 2 * d.se)));
  const make = useCallback((width: number) => Plot.plot({
    ...plotDefaults(width),
    height: 260,
    x: { label: "Predicted xP (average in each band)", domain: [0, Math.ceil(max)], grid: true },
    y: { label: "Actual points (average)", domain: [0, Math.ceil(max)], grid: true },
    marks: [
      Plot.line([[0, 0], [max, max]], { stroke: color.muted, strokeDasharray: "4,4", strokeWidth: 1 }),
      Plot.ruleX(data, { x: "predicted", y1: (d: Row) => d.actual - 2 * d.se, y2: (d: Row) => d.actual + 2 * d.se, stroke: color.s1, strokeWidth: 2 }),
      Plot.dot(data, { x: "predicted", y: "actual", r: 5, fill: color.s1, stroke: color.surface, strokeWidth: 2 }),
      Plot.tip(data, Plot.pointer({ x: "predicted", y: "actual",
        title: (d: Row) => `xP band ${d.bin}\npredicted ${pts(d.predicted)} · actual ${pts(d.actual)}\n${d.n.toLocaleString()} player-matches` })),
    ],
  }), [data, max]);
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Does xP mean what it says?</h3>
      <Chart make={make} height={260} ariaLabel="Calibration: predicted xP against actual points" />
      <p className="note">Players grouped by their xP. On the dashed line, a player given 4 xP scored 4 on average. The line through each dot shows how far that average could move by chance: the more players in a group, the shorter it is.</p>
    </div>
  );
}

function Tuning({ tuning }: { tuning: any }) {
  const chosen = Object.entries(tuning.chosen ?? {});
  if (!chosen.length) return null;
  return (
    <>
      <h3>How the team selection settings were chosen</h3>
      <p className="note" style={{ marginTop: 0 }}>
        Each setting was picked by replaying whole seasons ({(tuning.seasons ?? []).join(", ")}) week by week and keeping whichever
        scored the most points: {chosen.map(([k, val]) => `${k.replace(/_/g, " ")} ${String(val)}`).join(", ")}.
      </p>
    </>
  );
}
