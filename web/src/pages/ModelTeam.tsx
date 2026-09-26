import * as Plot from "@observablehq/plot";
import { useCallback, useMemo, useState } from "react";
import { color } from "../colors";
import { Pitch } from "../components/Pitch";
import { CaptainOddsTable, ChipOddsList, ScoreChart, band } from "../components/Simulation";
import { Chart, barPadding, Legend, Loading, Note, Table, Tiles, plotDefaults, type Column } from "../components/ui";
import type { ModelTeam, ModelWeek } from "../data";
import { int, money, pts, signed, when } from "../format";
import { totals, useAllGameweeks } from "../season";
import { useData, useSite } from "../site";

const CHIP_NAMES: Record<string, string> = { wildcard: "Wildcard", freehit: "Free Hit", bboost: "Bench Boost", "3xc": "Triple Captain" };
const SOURCE: Record<ModelWeek["source"], string> = {
  live: "Live", replay: "Replay", carried: "Carried over",
};

type Week = ModelWeek & { average: number | null; highest: number | null };

/** Did the week's score (before hits) land inside its simulated middle 80%? */
const inside = (w: ModelWeek) => w.simulation && w.gross !== undefined
  ? w.gross >= w.simulation.points.p10 && w.gross <= w.simulation.points.p90 : undefined;

export default function ModelTeamPage() {
  const site = useSite();
  const data = useData<ModelTeam>("modelteam.json");
  const all = useAllGameweeks(site.meta.played);
  const name = (id: number) => site.player.get(id)?.web_name ?? String(id);

  const weeks: Week[] = useMemo(() => (data?.gameweeks ?? []).map((w) => {
    const ev = site.meta.events.find((e) => e.id === w.gw);
    return { ...w, average: ev?.average ?? null, highest: ev?.highest ?? null };
  }), [data, site]);
  const choices = useMemo(() => [...(data?.next ? [data.next.gw] : []), ...weeks.map((w) => w.gw).reverse()], [data, weeks]);
  const [chosen, setChosen] = useState<number | null>(null);
  const gw = chosen ?? choices[0];

  const chart = useCallback((width: number) => Plot.plot({
    ...plotDefaults(width),
    height: 240,
    x: { label: null, type: "band", padding: barPadding(width, weeks.length), tickFormat: (d: number) => `GW${d}` },
    y: { label: "Points", grid: true, nice: true },
    marks: [
      Plot.ruleY([0], { stroke: color.grid }),
      Plot.barY(weeks, { x: "gw", y: "points", fill: color.s1, fillOpacity: (w: Week) => (w.source === "live" ? 1 : 0.45), ry2: 4 }),
      Plot.line(weeks, { x: "gw", y: "average", stroke: color.s2, strokeWidth: 2 }),
      Plot.dot(weeks, { x: "gw", y: "average", fill: color.s2, r: 4, stroke: color.surface, strokeWidth: 2 }),
      Plot.ruleX(weeks.filter((w) => w.simulation), { x: "gw", y1: (w: Week) => w.simulation!.points.p10,
        y2: (w: Week) => w.simulation!.points.p90, stroke: color.ink2, strokeWidth: 2 }),
      Plot.tickY(weeks.filter((w) => w.forecast != null), { x: "gw", y: "forecast", stroke: color.s3, strokeWidth: 3 }),
      Plot.tip(weeks, Plot.pointerX({ x: "gw", y: "points",
        title: (w: Week) => `GW${w.gw} (${SOURCE[w.source].toLowerCase()}): ${w.points} points, FPL average ${w.average ?? "–"}` +
          (w.forecast != null ? `\nforecast ${pts(w.forecast)}, scored ${w.gross} before transfer hits` : "") +
          (w.simulation ? `\nlikely range ${band(w.simulation.points.p10, w.simulation.points.p90)} (simulated)` : "") +
          (w.hits ? `\nincl. −${4 * w.hits} for transfers` : "") })),
    ],
  }), [weeks]);

  if (data === undefined) return <Loading />;
  if (data === null || (!weeks.length && !data.next)) {
    return <p>The Model's Team hasn't made a decision yet. Run <code>xpfpl recommend</code> (or <code>xpfpl modelteam</code>) before a deadline.</p>;
  }

  const total = weeks.reduce((s, w) => s + (w.points ?? 0), 0);
  const average = weeks.reduce((s, w) => s + (w.average ?? 0), 0);
  const above = weeks.filter((w) => w.average !== null && (w.points ?? 0) > w.average).length;
  const moves = weeks.filter((w) => w.chip !== "wildcard" && w.chip !== "freehit").reduce((s, w) => s + w.transfers.length, 0);
  const hits = weeks.reduce((s, w) => s + w.hits, 0);
  const missed = weeks.reduce((s, w) => s + ((w.best ?? 0) - (w.gross ?? 0)), 0);
  const replayed = weeks.filter((w) => w.source === "replay").map((w) => w.gw);
  const forecastWeeks = weeks.filter((w) => w.forecast != null);
  const forecastTotal = forecastWeeks.reduce((s, w) => s + w.forecast!, 0);
  const forecastScored = forecastWeeks.reduce((s, w) => s + (w.gross ?? 0), 0);
  const simulated = weeks.filter((w) => inside(w) !== undefined);
  const landed = simulated.filter((w) => inside(w)).length;

  const week = gw === data.next?.gw ? data.next : weeks.find((w) => w.gw === gw);
  const played = week && week.points !== undefined;
  const scores = played && all?.get(week.gw) ? totals(all.get(week.gw)!) : undefined;

  const columns: Column<Week>[] = [
    { key: "gw", label: "GW", numeric: true, value: (w) => w.gw },
    { key: "source", label: "Decision", value: (w) => w.source, render: (w) => <span className={w.source === "live" ? "tag" : "muted"}>{SOURCE[w.source]}</span>,
      title: "Live: saved before the deadline. Replay: filled in by the backtest for the weeks before the live record began." },
    { key: "points", label: "Points", numeric: true, value: (w) => w.points, render: (w) => <strong>{w.points}</strong> },
    { key: "forecast", label: "Forecast", numeric: true, value: (w) => w.forecast ?? null, render: (w) => pts(w.forecast),
      title: "What the model expected the team to score, before the deadline: captain doubled, bench only with Bench Boost, before transfer hits" },
    { key: "vsf", label: "vs forecast", numeric: true, value: (w) => (w.forecast == null ? null : (w.gross ?? 0) - w.forecast),
      render: (w) => (w.forecast == null ? "–" : <span className={(w.gross ?? 0) >= w.forecast ? "good" : "bad"}>{signed((w.gross ?? 0) - w.forecast, 1)}</span>),
      title: "Points before transfer hits, minus the forecast" },
    { key: "range", label: "Likely range", numeric: true, value: (w) => w.simulation?.points.p90 ?? null,
      render: (w) => !w.simulation ? "–"
        : <span className={inside(w) ? "good" : "bad"}>{band(w.simulation.points.p10, w.simulation.points.p90)}</span>,
      title: "The middle 80% of the team's simulated scores before the deadline (Monte Carlo): green if the score landed inside" },
    { key: "avg", label: "FPL average", numeric: true, value: (w) => w.average },
    { key: "vs", label: "vs average", numeric: true, value: (w) => (w.average === null ? null : (w.points ?? 0) - w.average),
      render: (w) => (w.average === null ? "–" : <span className={(w.points ?? 0) >= w.average ? "good" : "bad"}>{signed((w.points ?? 0) - w.average, 0)}</span>) },
    { key: "captain", label: "Captain", value: (w) => name(w.captain_played ?? w.captain),
      render: (w) => w.captain_played === null ? <span className="muted">none played</span>
        : <>{name(w.captain_played ?? w.captain)}{w.captain_played !== undefined && w.captain_played !== w.captain ? " (vice)" : ""}</> },
    { key: "transfers", label: "Transfers", value: (w) => w.transfers.length,
      render: (w) => w.transfers.length ? w.transfers.map((t) => `${name(t.out)} → ${name(t.in)}`).join(", ")
        : <span className="muted">{w.gw === weeks[0]?.gw && w.source !== "carried" ? "opening squad" : "–"}</span> },
    { key: "hits", label: "Hits", numeric: true, value: (w) => w.hits, render: (w) => (w.hits ? `−${4 * w.hits}` : "–") },
    { key: "chip", label: "Chip", value: (w) => w.chip, render: (w) => (w.chip ? <span className="tag">{CHIP_NAMES[w.chip] ?? w.chip}</span> : "–") },
    { key: "best", label: "Hindsight best", numeric: true, value: (w) => w.best, render: (w) => int(w.best),
      title: "The best XI and captain from the same 15 players, knowing the points" },
  ];

  return (
    <>
      <h2>The Model's Team</h2>
      <p className="lede">
        An FPL team run entirely by the model. Before every deadline it chooses the transfers, starting XI, bench order,
        captain and chips, and the decision is saved. After the gameweek it's scored like any other FPL team, with auto-subs,
        the vice-captain, chips and transfer hits all counted.
      </p>
      {weeks.length > 0 && (
        <Tiles tiles={[
          { label: "Total points", value: int(total), note: `${weeks.length} gameweek${weeks.length === 1 ? "" : "s"}` },
          { label: "Against the FPL average", value: signed(total - average, 0), note: `above average in ${above} of ${weeks.length}` },
          { label: "Transfers", value: int(moves), note: hits ? `${hits} hit${hits === 1 ? "" : "s"} (−${4 * hits})` : "no hits taken" },
          { label: "Left on the table", value: int(missed), note: "best XI and captain from the same 15, all season" },
          ...(forecastWeeks.length ? [{ label: "Against the forecast", value: signed(forecastScored - forecastTotal, 0),
            note: `forecast ${int(forecastTotal)}, scored ${int(forecastScored)} before hits (${forecastWeeks.length} GWs)` }] : []),
          ...(simulated.length ? [{ label: "Inside the likely range", value: `${landed} of ${simulated.length}`,
            note: "weeks whose score landed in the simulated middle 80% (about 4 in 5 should)" }] : []),
        ]} />
      )}

      {weeks.length > 0 && (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Points per gameweek</h3>
          <Legend items={[{ label: "Live", color: color.s1 }, ...(replayed.length ? [{ label: "Replay", color: `color-mix(in srgb, ${color.s1} 45%, transparent)` }] : []),
                          { label: "FPL average", color: color.s2, kind: "line" as const },
                          ...(forecastWeeks.length ? [{ label: "Forecast before the deadline", color: color.s3, kind: "line" as const }] : []),
                          ...(weeks.some((w) => w.simulation) ? [{ label: "Likely range (simulated)", color: color.ink2, kind: "line" as const }] : [])]} />
          <Chart make={chart} height={240} ariaLabel="The Model's Team's points per gameweek against the FPL average" />
        </div>
      )}

      <h3>The team</h3>
      <div className="toolbar">
        <label>
          Gameweek
          <select value={gw} onChange={(e) => setChosen(Number(e.target.value))}>
            {choices.map((g) => <option key={g} value={g}>GW{g}{g === data.next?.gw ? " (next)" : ""}</option>)}
          </select>
        </label>
        {week?.chip && <span className="tag">{CHIP_NAMES[week.chip] ?? week.chip}</span>}
        {week && <span className="muted">{SOURCE[week.source]}{week.source === "live" ? `, decided ${when(week.made_at)}` : ""}</span>}
      </div>
      {week && (
        <div className="card">
          <p style={{ marginTop: 0 }}>
            {played ? <><strong>{week.points}</strong> points{week.hits ? ` (after −${4 * week.hits} for transfers)` : ""}.
              {week.forecast != null && <> The forecast before the deadline was <strong>{pts(week.forecast)}</strong>, so it
                scored <span className={(week.gross ?? 0) >= week.forecast ? "good" : "bad"}>{signed((week.gross ?? 0) - week.forecast, 1)}</span> against it
                {week.hits ? " (before the transfer hits)" : ""}.</>}
              {" "}With hindsight, the best XI and captain from the same 15 would have scored <strong>{int(week.best)}</strong>.</>
              : <>Forecast: <strong>{pts(week.forecast ?? week.xp)}</strong> points (xP, captain doubled{week.chip === "bboost" ? ", bench included" : ""}).
                This can still change until the deadline, {when(site.meta.next_deadline)}.</>}
            {" "}{week.transfers.length ? `Transfers: ${week.transfers.map((t) => `${name(t.out)} (${money(t.sold)}) → ${name(t.in)} (${money(t.bought)})`).join(", ")}.`
              : week.source === "carried" ? "No decision was saved this week, so last week's team played on." : "No transfers."}
            {" "}{money(week.bank)} in the bank and {week.free_transfers} free transfer{week.free_transfers === 1 ? "" : "s"} before the week.
          </p>
          <Pitch gw={week.gw} lineup={week.lineup} bench={week.bench} captain={week.captain} vice={week.vice}
                 marked={played ? new Set(week.best_xi) : undefined} flags={!played}
                 line={(p) => {
                   const player = site.player.get(p);
                   if (!played) return <><b>{pts(player?.forecast)}</b> xP<span className="xp-extra"> · £{player?.now_cost.toFixed(1)}</span></>;
                   const g = scores?.get(p);
                   const multiplier = p === week.captain_played ? (week.chip === "3xc" ? 3 : 2) : 1;
                   return <><b>{(g?.points ?? 0) * multiplier}</b> pts<span className="xp-extra"> · {pts(g?.xp)} xP</span></>;
                 }} />
          {!!week.autosubs?.length && (
            <p className="note">Auto-subs: {week.autosubs.map(([out, sub]) => `${name(sub)} on for ${name(out)}`).join(", ")}.</p>
          )}
          <p className="note">
            {played ? "The captain's points are doubled (tripled with Triple Captain); bench points count only with Bench Boost. A dashed outline marks a player the hindsight-best XI would have started."
              : `xP is each player's expected points for GW${week.gw}. The fixture is shaded by FPL's difficulty rating, and a number in the corner is FPL's chance of playing.`}
          </p>
        </div>
      )}

      {week?.simulation && (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>How GW{week.gw} could go</h3>
          <p style={{ marginTop: 0 }}>
            Played {week.simulation.sims.toLocaleString()} times before the deadline, the team scored{" "}
            <strong>{band(week.simulation.points.p10, week.simulation.points.p90)}</strong> in the middle 80% of simulated weeks
            (median {Math.round(week.simulation.points.p50)}, average {pts(week.simulation.points.mean)} with auto-subs and the vice-captain).
            {played && week.gross !== undefined && <> It scored <strong>{week.gross}</strong> before transfer hits: {inside(week)
              ? "inside that range." : week.gross > week.simulation.points.p90 ? "above it, a one-in-ten week." : "below it, a one-in-ten week."}</>}
          </p>
          <ScoreChart spread={week.simulation.points} forecast={week.forecast ?? week.xp} actual={played ? week.gross : null} gw={week.gw} />
          <h4>Captain options</h4>
          <CaptainOddsTable sim={week.simulation} captain={week.captain} />
          <p className="note">
            Each option's own points in the simulations (before the armband). Best pick: how often he outscored every other
            option in the same simulated week. The captain is still the highest xP; these odds show how close the call was.
          </p>
          {Object.keys(week.simulation.chips).length > 0 && <>
            <h4>Chips</h4>
            <ChipOddsList chips={week.simulation.chips} />
          </>}
          <p className="note">
            Each simulated week draws the goals in every fixture, who plays, goals, assists, clean sheets, bonus and cards, with
            each player's average matched to his xP. See <a href="#about">About</a> for how, and <a href="#accuracy">Model
            Accuracy</a> for how often the simulated chances came true. By design, one week in ten falls below the range
            and one in ten above it.
          </p>
        </div>
      )}

      {weeks.length > 0 && (
        <>
          <h3>Week by week</h3>
          <Table columns={columns} data={[...weeks].reverse()} rowKey={(w) => w.gw} />
          <dl className="defs">
            <dt>Live</dt>
            <dd>Decided and saved before the gameweek's deadline, using only information available at the time.</dd>
            <dt>Replay</dt>
            <dd>
              Worked out afterwards for the gameweeks before the team began, using a model trained only on earlier seasons
              and each player's form at the time. Injury news isn't included, so treat these weeks as a guide. They're shown
              paler in the chart.
            </dd>
            <dt>Forecast</dt>
            <dd>
              What the model expected the team to score, made before the deadline and counted the way the week is scored:
              captain doubled (tripled with Triple Captain), bench only with Bench Boost, before transfer hits. For a
              carried-over week it's summed from that week's player forecasts. It's drawn as a line on each week's bar.
            </dd>
            <dt>Likely range</dt>
            <dd>
              The middle 80% of the team's scores when the week was simulated thousands of times before the deadline (Monte
              Carlo). About four weeks in five should land inside it; green if this one did.
            </dd>
            <dt>Carried over</dt>
            <dd>No decision was saved before the deadline, so the previous week's team played on and a free transfer was banked.</dd>
          </dl>
        </>
      )}

      <h3>Chips</h3>
      <table>
        <thead><tr><th>Chip</th><th>Window</th><th>Played</th></tr></thead>
        <tbody>
          {[...site.meta.chips].sort((a, b) => a.start - b.start || (CHIP_NAMES[a.name] ?? a.name).localeCompare(CHIP_NAMES[b.name] ?? b.name)).map((c) => {
            const used = [...weeks, ...(data.next ? [data.next] : [])].find((w) => w.chip === c.name && w.gw >= c.start && w.gw <= c.stop);
            return (
              <tr key={`${c.name}-${c.start}`}>
                <td>{CHIP_NAMES[c.name] ?? c.name}</td>
                <td>GW{c.start}–{c.stop}</td>
                <td>{used ? `GW${used.gw}` : <span className="muted">Not yet</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <Note>
        {replayed.length > 0 && <>
          GW{replayed.length === 1 ? replayed[0] : `${replayed[0]}–${replayed[replayed.length - 1]}`} {replayed.length === 1 ? "is a replay" : "are replays"}:
          the live record began later.{" "}
        </>}
        The model is <strong>{data.model}</strong>. It starts with £100m and pays FPL's selling prices, so it can't profit from rises it didn't hold.
      </Note>
    </>
  );
}
