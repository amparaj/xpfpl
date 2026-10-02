// The technical report: how the system is built and tested, section by section, with the charts
// drawn from the exported reports (accuracy.json) so every number stays current. Reached from
// About (#about/report, or #about/report/<section> to open at a section).

import * as Plot from "@observablehq/plot";
import { useCallback, useEffect, type ReactNode } from "react";
import { color } from "../colors";
import { Flow } from "../components/Flow";
import { PAPER } from "../components/Simulation";
import { Chart, Legend, Loading, Tiles, plotDefaults } from "../components/ui";
import type { Accuracy, Row } from "../data";
import { dec, int, pct } from "../format";
import { useData, useHash, useSite } from "../site";
import { Calibration } from "./Accuracy";

const SECTIONS = [
  { id: "summary", title: "Summary" },
  { id: "architecture", title: "System architecture" },
  { id: "data", title: "Data" },
  { id: "inputs", title: "What this model reads" },
  { id: "models", title: "The models" },
  { id: "validation", title: "Validation methodology" },
  { id: "accuracy", title: "Forecast accuracy" },
  { id: "simulation", title: "Simulation" },
  { id: "selection", title: "Team selection" },
  { id: "limitations", title: "Limitations" },
  { id: "reproduce", title: "Reproducibility" },
] as const;

function Section({ id, n, children }: { id: string; n: number; children: ReactNode }) {
  const title = SECTIONS.find((s) => s.id === id)!.title;
  return (
    <section id={`report-${id}`} className="report-section">
      <h3><span className="report-n">{n}</span>{title}</h3>
      {children}
    </section>
  );
}

/** Seasons as "2016-17" from `first` to `last`. */
function seasonRange(first: string, last: string): string[] {
  const a = Number(first.slice(0, 4)), b = Number(last.slice(0, 4));
  return Array.from({ length: b - a + 1 }, (_, i) => `${a + i}-${String((a + i + 1) % 100).padStart(2, "0")}`);
}
const before = (s: string, by = 1) => seasonRange(`${Number(s.slice(0, 4)) - by}`, `${Number(s.slice(0, 4)) - by}`)[0];

type Role = "train" | "stop" | "test" | "live" | null;

/** The validation design as a grid of seasons: what each test trained on, stopped early on and was scored on. */
function Timeline({ test, walk, current }: { test?: string; walk: string[]; current: string }) {
  const seasons = seasonRange("2016-17", current);
  const row = (t: string): Role[] => seasons.map((s) => (s === t ? "test" : s === before(t) ? "stop" : s < t ? "train" : null));
  const groups: { title: string; rows: { label: string; roles: Role[] }[] }[] = [
    ...(test ? [{ title: "Held-out season: picks the model", rows: [{ label: test, roles: row(test) }] }] : []),
    ...(walk.length ? [{ title: "Walk-forward: one season at a time", rows: walk.map((t) => ({ label: t, roles: row(t) })) }] : []),
    { title: "Live: this season", rows: [{ label: current, roles: seasons.map((s): Role => (s === current ? "live" : s < current ? "train" : null)) }] },
  ];
  return (
    <div className="timeline-wrap">
      <Legend items={[
        { label: "Trained on", color: `color-mix(in srgb, ${color.s1} 45%, transparent)` },
        { label: "Early stopping (then trained on)", color: color.neutral },
        { label: "Scored on, never seen", color: color.s2 },
        { label: "Forecast before each deadline", color: color.s3 },
      ]} />
      <div className="timeline" role="table" aria-label="Which seasons each test trained on and was scored on"
           style={{ ["--cols" as string]: seasons.length }}>
        <div className="timeline-row timeline-head" role="row">
          <span role="columnheader" />
          {seasons.map((s) => <span key={s} role="columnheader">{s.slice(2)}</span>)}
        </div>
        {groups.map((g) => [
          <div key={g.title} className="timeline-group" role="row"><span role="rowheader">{g.title}</span></div>,
          ...g.rows.map((r) => (
            <div key={`${g.title}-${r.label}`} className="timeline-row" role="row">
              <span role="rowheader" className="timeline-label">{r.label}</span>
              {r.roles.map((role, i) => (
                <span key={i} role="cell" className={`timeline-cell${role ? ` tl-${role}` : ""}`}
                      title={role ? `${seasons[i]}: ${{ train: "trained on", stop: "early stopping", test: "scored on", live: "live forecasts" }[role]}` : undefined} />
              ))}
            </div>
          )),
        ])}
      </div>
    </div>
  );
}

const COMPARE_COLORS = () => ({ ensemble: color.s1, Baseline: color.s2, baseline: color.s2, "FPL xP": color.s3 });

export default function Report() {
  const site = useSite();
  const report = useData<Accuracy>("accuracy.json");
  const hash = useHash();
  const target = hash.split("/")[2];

  // #about/report/<section>: open at that section once the page has drawn.
  useEffect(() => {
    if (!target || report === undefined) return;
    document.getElementById(`report-${target}`)?.scrollIntoView({ block: "start" });
  }, [target, report]);

  const v = report?.validation, c = report?.comparison, r = report?.robustness, t = report?.tuning;
  const model = site.meta.model;
  const models: Row[] = c?.models ?? [];
  const ceiling = c?.ceiling;
  const headline = (name: string) => (v?.headline ?? []).find((h: Row) => h.model === name && h.subset === "Players getting minutes");
  const ours = headline(v?.primary ?? model), base = headline(v?.reference ?? "Baseline");
  const spread = (r?.spread ?? []).find((x: Row) => x.model === "ensemble");
  const points: Record<string, Record<string, number>> = r?.backtest?.points ?? {};
  const seasonMean = (variant: string) => {
    const xs = Object.values(points[variant] ?? {});
    return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : undefined;
  };
  const sim = v?.simulation;
  const horizon = t?.chosen?.horizon ?? 3;

  // ---------------------------------------------------------------- charts

  const modelChart = useCallback((width: number) => {
    const data = [...models].sort((a, b) => a.rmse - b.rmse);
    const isOurs = (d: Row) => d.model === model;
    return Plot.plot({
      ...plotDefaults(width),
      marginLeft: 92,
      height: data.length * 26 + 50,
      x: { label: "RMSE, points per player-match (lower is better)", grid: true, nice: true },
      y: { label: null, domain: data.map((d) => d.model) },
      marks: [
        ...(ceiling ? [Plot.rectX([ceiling], { x1: "rmse_p5", x2: "rmse_p95", fill: color.grid, fillOpacity: 0.8 })] : []),
        Plot.ruleY(data, { y: "model", x1: () => Math.min(...data.map((x) => x.rmse)) - 0.05, x2: "rmse", stroke: color.grid }),
        Plot.dot(data, { x: "rmse", y: "model", r: 6, fill: (d: Row) => (isOurs(d) ? color.s1 : color.muted), stroke: color.surface, strokeWidth: 2 }),
        Plot.text(data, { x: "rmse", y: "model", text: (d: Row) => dec(d.rmse, 3), dx: 10, textAnchor: "start", fill: color.ink2 }),
        Plot.tip(data, Plot.pointerY({ x: "rmse", y: "model", title: (d: Row) =>
          `${d.model}: ${d.description}\nRMSE ${dec(d.rmse, 3)} · MAE ${dec(d.mae, 3)} · rank corr. ${dec(d.spearman, 3)}` })),
      ],
    });
  }, [models, ceiling, model]);

  const horizons: Row[] = (c?.horizons ?? []).filter((h: Row) => h.model !== "FPL xP");
  const horizonChart = useCallback((width: number) => {
    const others = horizons.filter((h) => h.model !== model);
    const mine = horizons.filter((h) => h.model === model);
    return Plot.plot({
      ...plotDefaults(width),
      height: 240,
      marginRight: 70,
      x: { label: "Gameweeks ahead", ticks: [1, 2, 3], tickFormat: (d: number) => `${d}` },
      y: { label: "RMSE (points)", grid: true, nice: true },
      marks: [
        Plot.line(others, { x: "horizon", y: "rmse", z: "model", stroke: color.muted, strokeWidth: 1, strokeOpacity: 0.6 }),
        Plot.line(mine, { x: "horizon", y: "rmse", stroke: color.s1, strokeWidth: 2 }),
        Plot.dot(mine, { x: "horizon", y: "rmse", fill: color.s1, r: 4, stroke: color.surface, strokeWidth: 2 }),
        Plot.text(mine.slice(-1), { x: "horizon", y: "rmse", text: () => model, dx: 8, textAnchor: "start", fill: color.ink }),
        Plot.text(others.filter((h) => h.horizon === 3 && h.model === "baseline"), { x: "horizon", y: "rmse", text: () => "baseline", dx: 8, textAnchor: "start", fill: color.muted }),
        Plot.tip(horizons, Plot.pointer({ x: "horizon", y: "rmse", title: (h: Row) => `${h.model}, ${h.horizon} ahead\nRMSE ${dec(h.rmse, 3)} · MAE ${dec(h.mae, 3)}` })),
      ],
    });
  }, [horizons, model]);

  const positions: Row[] = v?.by_position ?? [];
  const positionModels: string[] = v?.models ?? [];
  const positionChart = useCallback((width: number) => {
    const colors = COMPARE_COLORS() as Record<string, string>;
    return Plot.plot({
      ...plotDefaults(width),
      height: 240,
      fx: { label: null, domain: ["GKP", "DEF", "MID", "FWD"] },
      x: { axis: null, domain: positionModels, paddingInner: 0.15 },
      y: { label: "RMSE (points)", grid: true, nice: true },
      color: { domain: positionModels, range: positionModels.map((m) => colors[m] ?? color.neutral) },
      marks: [
        Plot.barY(positions, { fx: "position", x: "model", y: "rmse", fill: "model", ry2: 4, insetLeft: 1, insetRight: 1 }),
        Plot.ruleY([0], { stroke: color.grid }),
        Plot.tip(positions, Plot.pointer({ fx: "position", x: "model", y: "rmse", title: (d: Row) =>
          `${d.position} · ${d.model}\nRMSE ${dec(d.rmse, 2)} · bias ${dec(d.bias, 2)}\n${int(d.n)} player-matches` })),
      ],
    });
  }, [positions, positionModels]);

  const errors: Row[] = [...(v?.errors ?? [])].sort((a: Row, b: Row) => a.order - b.order);
  const errorChart = useCallback((width: number) => Plot.plot({
    ...plotDefaults(width),
    height: 230,
    marginBottom: 56,
    x: { label: "Points scored minus xP", domain: errors.map((e) => e.band), tickRotate: width < 560 ? -40 : 0 },
    y: { label: "Share of player-matches", grid: true, tickFormat: (d: number) => `${Math.round(d * 100)}%` },
    marks: [
      Plot.barY(errors, { x: "band", y: "share", fill: (e: Row) => (e.band === "within 1" ? color.s1 : `color-mix(in srgb, ${color.s1} 45%, transparent)`), ry2: 4, insetLeft: 1, insetRight: 1 }),
      Plot.ruleY([0], { stroke: color.grid }),
      Plot.tip(errors, Plot.pointerX({ x: "band", y: "share", title: (e: Row) => `${e.band}: ${pct(e.share, 1)} (${int(e.count)})` })),
    ],
  }), [errors]);

  const replaySeasons: string[] = r?.seasons ?? [];
  const replay = ["ensemble", "baseline"].flatMap((variant) =>
    replaySeasons.filter((s) => points[variant]?.[s] !== undefined).map((s) => ({ season: s, variant, label: variant === "ensemble" ? "This model" : "5-match average", points: points[variant][s] })));
  const replayChart = useCallback((width: number) => Plot.plot({
    ...plotDefaults(width),
    height: 240,
    marginLeft: 50,
    x: { label: null, type: "point", padding: 0.3 },
    y: { label: "FPL points in the season", grid: true, nice: true },
    color: { domain: ["This model", "5-match average"], range: [color.s1, color.s2] },
    marks: [
      Plot.line(replay, { x: "season", y: "points", stroke: "label", strokeWidth: 2 }),
      Plot.dot(replay, { x: "season", y: "points", fill: "label", r: 4, stroke: color.surface, strokeWidth: 2 }),
      Plot.tip(replay, Plot.pointer({ x: "season", y: "points", title: (d: { season: string; label: string; points: number }) => `${d.label}, ${d.season}: ${int(d.points)} points` })),
    ],
  }), [replay]);

  if (report === undefined) return <Loading />;

  let n = 0;
  return (
    <article className="report">
      <p className="lede">
        How xP-FPL is built and tested: the data, what this {model} model reads, how it is validated, how accurate it is and how
        the team is picked. Every number on this page is read from the latest reports when the site is published.
      </p>
      <nav className="report-toc" aria-label="Sections">
        {SECTIONS.map((s, i) => <a key={s.id} href={`#about/report/${s.id}`}>{i + 1}. {s.title}</a>)}
      </nav>

      <Section id="summary" n={++n}>
        <Tiles tiles={[
          ...(ours ? [{ label: "Forecast error", value: `${dec(ours.rmse, 2)} RMSE`, note: `on ${v.season}, never seen in training${base ? `; ${dec(base.rmse, 2)} for a 5-match average` : ""}` }] : []),
          ...(ours ? [{ label: "Ranking skill", value: dec(ours.spearman, 2), note: `rank correlation with the points scored${base ? ` (5-match average ${dec(base.spearman, 2)})` : ""}` }] : []),
          ...(ceiling ? [{ label: "Room left", value: `${dec(ceiling.rmse_median, 2)}`, note: "RMSE of a model that knew every player's true chances: the luck no model can remove" }] : []),
          ...(spread ? [{ label: `Across ${spread.seasons} seasons`, value: `${dec(spread.rmse_mean, 2)} ± ${dec(spread.rmse_sd, 2)}`, note: "RMSE, each season forecast by a model trained only on earlier ones" }] : []),
          ...(seasonMean("ensemble") !== undefined ? [{ label: "Replayed seasons", value: `${int(seasonMean("ensemble"))} pts`, note: `a season on average, against ${int(seasonMean("baseline"))} picking by recent form` }] : []),
          ...(sim ? [{ label: "Simulated ranges", value: pct(sim.coverage_80), note: "of real scores inside the middle-80% range (80% is right)" }] : []),
        ]} />
        <p>
          Accuracy is measured on players who got on the pitch: about half of all player-matches are players who didn't play, and
          they are easy to forecast. RMSE (root mean squared error) punishes big misses more than small ones; rank correlation asks
          whether the players expected to score most did score most, which is what picking a team needs.
        </p>
      </Section>

      <Section id="architecture" n={++n}>
        <p>Data comes in at the top and a team comes out at the bottom. Click a box for its section.</p>
        <Flow detailed horizon={horizon} />
      </Section>

      <Section id="data" n={++n}>
        <div className="table-wrap">
          <table className="compact report-table">
            <thead><tr><th>Source</th><th>What</th><th>Seasons</th></tr></thead>
            <tbody>
              <tr><td>FPL's API</td><td>this season: every player's matches, prices, ownership, transfers, injury flags, fixtures</td><td>{site.meta.season}</td></tr>
              <tr><td>vaastav's Fantasy-Premier-League</td><td>every past season's player-match rows (FPL's own data, kept on GitHub)</td><td>2016-17 on</td></tr>
              <tr><td>Polymarket</td><td>match result, goals and goalscorer prices at each FPL deadline</td><td>2024-25 on</td></tr>
              <tr><td>Football-Data.co.uk</td><td>bookmakers' match odds (to judge the market and the ratings; not a model input)</td><td>2016-17 on</td></tr>
              <tr><td>FPL-Core-Insights</td><td>cup and European fixtures and minutes; match data (shots, xG) for the match pages</td><td>2025-26 on</td></tr>
            </tbody>
          </table>
        </div>
        <p>
          About a quarter of a million player-matches in all, one row per player per fixture, players linked across seasons by FPL's
          permanent player code (FPL renumbers its ids every season). Every source is kept in a compressed archive in the project's
          repository, so a forecast can always be rebuilt from exactly what was known at the time (<a href="#data">Data</a>).
          {v && <> The held-out test season, {v.season}, has {int(v.rows)} player-matches, {int(v.active_rows)} of them with minutes.</>}
        </p>
      </Section>

      <Section id="inputs" n={++n}>
        <p>Every input is built only from matches before the deadline it forecasts. They fall into five groups:</p>
        <div className="pillars">
          <div className="pillar"><strong>Minutes and role</strong><p>Minutes, appearances and 60-minute games over the last 3, 5 and 10 matches;
            his price and minutes rank within his club (the first choice is the expensive one who plays). The minutes model
            turns this into chances of no minutes, a cameo or 60+.</p></div>
          <div className="pillar"><strong>Underlying numbers</strong><p>Rolling points, goals, assists, bonus and BPS rank over 3, 5 and 10 matches; points,
            xG, xA, FPL's threat and creativity, saves, clean sheets and defensive contribution per 90 minutes over the last 20 and
            38 matches, so a short purple patch doesn't dominate.</p></div>
          <div className="pillar"><strong>Fixture</strong><p>Club attack and defence ratings from a Poisson model refitted before every gameweek
            (half-weighted every 240 days, goals and xG mixed), giving each side's expected goals and clean-sheet chance;
            home or away; his points at that venue. Where Polymarket prices the match, its goal expectations are used, and
            the component model moves its clean-sheet and goals-conceded parts halfway to the market's.</p></div>
          <div className="pillar"><strong>Crowd</strong><p>Net transfers before the deadline and ownership. The single most useful group: it carries
            team news (a knock, a rotation hint) before the stats can.</p></div>
          <div className="pillar"><strong>After the forecast</strong><p>FPL's chance-of-playing flag (assumed to recover 25% a week), the midweek factor
            for players whose club played a cup or European match, and a cut to a tenth for anyone whose goalscorer odds are
            {" "}{pct(site.meta.out_threshold)} or less.</p></div>
        </div>
        <p className="note">
          The crowd inputs cut every model's error by about 0.05 RMSE, more than the gap between the best and worst architectures.
          Longer per-90 windows, club ratings, venue and BPS rank each added less than 0.005. Other ideas were tested and left out
          because they didn't help on seasons they weren't tuned on: club rotation rates, rest days, finishing luck, and bookmaker
          odds as inputs. A penalty-taker adjustment (FPL's penalty order against who took them lately) is recorded with every
          forecast but not applied: on 2025-26 it explained almost none of the error, so this season's results will decide it.
        </p>
      </Section>

      <Section id="models" n={++n}>
        <p>
          Several models are trained on the same rows: one per past player-match, its inputs as they stood before it, its target the
          FPL points scored. They are compared on the same held-out season, and the default is chosen by points in replayed
          seasons as well as forecast error.
          {model === "ensemble" && <> The default, <strong>ensemble</strong>, averages three: a neural network (PyTorch, two hidden
          layers of 128 and 64), gradient-boosted trees (LightGBM) and a minutes model (PyTorch) that predicts the chance of 0,
          1-59 and 60+ minutes and the points expected given each. Averaging different kinds of model cancels some of each one's
          mistakes.</>}
        </p>
        {models.length > 0 && (
          <div className="card">
            <h3 style={{ marginTop: 0 }}>Every model on {c.season}</h3>
            <Legend items={[{ label: model, color: color.s1, kind: "dot" }, { label: "Other models", color: color.muted, kind: "dot" },
              ...(ceiling ? [{ label: "Perfect-model range", color: color.grid }] : [])]} />
            <Chart make={modelChart} height={models.length * 26 + 50} ariaLabel="Forecast error of every model on the held-out season" />
            <p className="note">The shaded band is where a model that knew every player's true chances would land (5th to 95th percentile of
              simulated seasons). Hover a model for what it is.</p>
          </div>
        )}
        <p>
          Double gameweeks add both matches; blank gameweeks score zero. The neural networks are trained with early stopping: training
          stops when the error on a season it isn't trained on stops improving (see the next section), then it is refitted on
          everything.
        </p>
      </Section>

      <Section id="validation" n={++n}>
        <p>Three kinds of test, each scoring forecasts on matches the forecasting model never saw:</p>
        <ol>
          <li><strong>Held-out season.</strong> Every model is trained on the seasons before {v ? before(v.season) : "the last one"}, stopped
            early on {v ? before(v.season) : "the last one"} (then refitted including it), and scored once on {v?.season ?? "the most recent season"}.
            This picks which model the site uses.</li>
          <li><strong>Walk-forward.</strong> The same again for each of {replaySeasons.length || "six"} seasons{replaySeasons.length > 0 && <> ({replaySeasons[0]} to{" "}
            {replaySeasons[replaySeasons.length - 1]})</>}, each forecast and replayed week by week by a model trained only on the seasons
            before it. This shows whether a result holds up or was one season's luck.</li>
          <li><strong>Live.</strong> This season, each forecast is saved before the deadline and scored once the gameweek is played
            (<a href="#accuracy">This Model's Accuracy</a>). Nothing can be adjusted after the fact.</li>
        </ol>
        <Timeline test={v?.season} walk={replaySeasons} current={site.meta.season} />
        <p>
          <strong>No peeking.</strong> Every input on a training row comes from the matches before it. Inside a replay, all future
          gameweeks are forecast from one snapshot at the deadline, as a manager would have to. As a check, the inputs were rebuilt from
          data cut off at a deadline and compared with the full build: they matched, apart from rounding in the club-ratings fit.
        </p>
        <p>
          <strong>Replays.</strong> To test the team picking, a season is replayed one deadline at a time: forecast, pick, transfer,
          score the real points with auto-subs, carry the bank, free transfers and selling prices into the next week.
          {r?.backtest?.noise_sd && <> A replayed season moves by about {int(r.backtest.noise_sd)} points when the forecasts change by
          only 10%, so two settings closer than that can't be told apart.</>}
          {" "}Settings that pay off within a few weeks (captaincy, bench, chips, how far to look ahead) can be compared more
          precisely in pairs: the second setting takes over the first one's exact squad, bank and free transfers at every deadline
          for a few weeks, and the two are scored over the same weeks. Tested with the same 10% noise, that measured a setting's
          worth about five times more precisely than two full replays. It can't see what builds up over a season, such as squad
          value from price rises, so those are still judged on full replays.
        </p>
      </Section>

      <Section id="accuracy" n={++n}>
        {horizons.length > 0 && (
          <div className="card">
            <h3 style={{ marginTop: 0 }}>Forecasts fade with distance</h3>
            <Chart make={horizonChart} height={240} ariaLabel="Forecast error one, two and three gameweeks ahead" />
            <p className="note">Error forecasting 1, 2 and 3 gameweeks ahead on {c.season}: {model} in blue, other models in grey. Later weeks
              count for less when picking a team for this reason.</p>
          </div>
        )}
        <div className="grid-2">
          {positions.length > 0 && (
            <div className="card">
              <h3 style={{ marginTop: 0 }}>By position</h3>
              <Legend items={positionModels.map((m) => ({ label: m === "Baseline" ? "5-match average" : m, color: (COMPARE_COLORS() as Record<string, string>)[m] ?? color.neutral }))} />
              <Chart make={positionChart} height={240} ariaLabel="Forecast error by position" />
              <p className="note">Defenders and forwards are hardest: clean sheets and goals are all-or-nothing.</p>
            </div>
          )}
          {errors.length > 0 && (
            <div className="card">
              <h3 style={{ marginTop: 0 }}>How far off</h3>
              <Chart make={errorChart} height={230} ariaLabel="How far points scored were from xP" />
              <p className="note">Most forecasts are within a point. The long tail to the right is hauls: goals no forecast sees coming.</p>
            </div>
          )}
        </div>
        {v && <Calibration validation={v} />}
        <p>
          <strong>Confidence.</strong> Each player's forecast carries a High, Medium or Low confidence label: how far this model's three
          parts (the neural network, the gradient boosting and the minutes model) disagree on him, as their spread over his xP. On the
          six seasons above, among players on 2+ xP, the forecasts where they disagreed most (the Low 15%) missed by about 1.2 times the
          usual squared error for the same xP, in five of the six seasons; where they agreed (High, about half) a little less than usual.
          The cut-offs were fixed on 2020-21 to 2022-23 and held on the three later seasons. The chance of playing, the player's history
          and retraining from a different random start were weaker signals. Risk is separate: how often a pick scores 2 or fewer.
        </p>
        {r?.calibration && <p className="note">Pooled over the walk-forward seasons, points ≈ {dec(r.calibration.intercept, 2)} + {dec(r.calibration.slope, 2)} × xP: an
          xP of 6 means about 6. Recalibrating on earlier seasons gained nothing.</p>}
      </Section>

      <Section id="simulation" n={++n}>
        <p>
          After each forecast the coming gameweeks are played thousands of times. Each simulated week draws both sides' goals (from the
          market or the club ratings), each club's line-up (every player's minutes from the minutes model, drawn for the whole side
          at once so a club fields as many players as their chances add up to), his share of his side's goals and assists,
          clean sheets, saves, defensive actions, and bonus and cards as they have fallen in the past. Teammates share their side's
          goals, so they rise and fall together. Each player's simulated average is matched to his xP, so the simulation spreads the
          points without changing how many are expected.
        </p>
        {sim && <Tiles tiles={[
          { label: "Inside the middle 80%", value: pct(sim.coverage_80), note: `real scores on ${v.season} (80% is right)` },
          { label: "Club totals inside", value: pct(sim.club_coverage_80), note: "each club's total in each match (80% is right)" },
          { label: "Spread of points", value: `${dec(sim.spread.simulated, 1)} vs ${dec(sim.spread.actual, 1)}`, note: "variance, simulated against real" },
        ]} />}
        <p>
          The simulations give each player a range and his chances of 10+ and of 2 or fewer, the team's likely score, captain and chip
          odds. They don't change the picks: with FPL's scoring, the most expected points are the same either way, as{" "}
          <a href={PAPER}>Ramezani &amp; Dinh (2026)</a> also found.
        </p>
      </Section>

      <Section id="selection" n={++n}>
        <p>
          An integer program (PuLP with the CBC solver) picks the squad worth the most expected points over the next {horizon} gameweeks,
          later weeks discounted{t?.chosen?.discount !== undefined && <> by {dec(t.chosen.discount, 2)} a week</>}, subject to FPL's rules:
        </p>
        <ul>
          <li>15 players: 2 goalkeepers, 5 defenders, 5 midfielders, 3 forwards; no more than 3 from one club;</li>
          <li>the budget, using each owned player's selling price (half of any rise is kept, falls in full);</li>
          <li>a valid starting XI (at least 3 defenders, 2 midfielders, 1 forward), a captain and vice-captain;</li>
          <li>transfers: each one past the free transfers costs 4 points; a free transfer saved is given a value, since it can be banked.</li>
        </ul>
        <p>
          Bench players count for a little, and a small term for expected price rises breaks ties. Chips are played when this week's
          gain clears a threshold and no later week in the same chip window looks better.
        </p>
        {t?.config && (
          <div className="table-wrap">
            <table className="compact report-table">
              <thead><tr><th>Setting</th><th>Chosen by replaying seasons</th></tr></thead>
              <tbody>{(t.config as string[]).map((line) => {
                const [k, val] = line.split(" = ");
                return <tr key={k}><td><code>{k}</code></td><td className="num">{val}</td></tr>;
              })}</tbody>
            </table>
          </div>
        )}
        {replay.length > 0 && (
          <div className="card">
            <h3 style={{ marginTop: 0 }}>Points replaying each season</h3>
            <Legend items={[{ label: "This model", color: color.s1, kind: "line" }, { label: "5-match average", color: color.s2, kind: "line" }]} />
            <Chart make={replayChart} height={240} ariaLabel="FPL points in each replayed season, this model against picking by recent form" />
            <p className="note">Each season replayed from the first deadline by a model trained only on earlier seasons, with the same optimiser.
              {seasonMean("oracle") !== undefined && <> Perfect foresight of every result would have scored about {int(seasonMean("oracle"))} a season.</>}</p>
          </div>
        )}
      </Section>

      <Section id="limitations" n={++n}>
        <ul>
          <li>Football is noisy: most of a week's points can't be forecast by anyone, and a season's total swings by
            {r?.backtest?.noise_sd ? ` about ${int(r.backtest.noise_sd)}` : " a lot"} points on luck alone.</li>
          <li>Transfers made days before the deadline are only partly counted: the crowd input is complete only at the deadline.</li>
          <li>Rule changes take time to learn: defensive contribution points began in 2025-26, so defenders were under-forecast that season.</li>
          <li>Later weeks of a plan assume no new information; only this week's transfer is real.</li>
          <li>Price changes are estimated from form and transfers, not FPL's hidden thresholds.</li>
          <li>Free transfers are estimated by replaying the season; one-off top-ups aren't modelled.</li>
          <li>The settings were tuned on some of the same seasons they are reported on, which flatters them a little.</li>
        </ul>
      </Section>

      <Section id="reproduce" n={++n}>
        <p>
          The code{site.meta.repo && <> is <a href={site.meta.repo}>on GitHub</a> and</>} runs on an ordinary computer (CPU only, no GPU). Each week:
        </p>
        <pre className="code">{`xpfpl fetch      # the gameweek's data, into the archive
xpfpl markets    # betting odds at each deadline
xpfpl train      # retrain on everything up to now
xpfpl recommend  # forecast, simulate, pick; saves the forecast
xpfpl scorecard  # score last week's saved forecast
xpfpl publish    # this site`}</pre>
        <p>
          Occasional: <code>xpfpl compare</code> (every model on the held-out season), <code>xpfpl robustness</code> (the walk-forward
          seasons) and <code>xpfpl tune</code> (the settings, by replaying seasons). The archive behind every number is on the{" "}
          <a href="#data">Data</a> page.
        </p>
      </Section>
    </article>
  );
}
