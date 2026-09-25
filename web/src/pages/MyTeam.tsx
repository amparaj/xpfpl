import * as Plot from "@observablehq/plot";
import { useCallback, useMemo, useState } from "react";
import { color } from "../colors";
import { Chart, barPadding, Club, Legend, Loading, Note, Table, Tiles, plotDefaults, type Column } from "../components/ui";
import type { Manager } from "../data";
import { POSITIONS, compact, int, money, pts, signed } from "../format";
import { totals, useAllGameweeks } from "../season";
import { useData, useSite } from "../site";

const CHIP_NAMES: Record<string, string> = { wildcard: "Wildcard", freehit: "Free Hit", bboost: "Bench Boost", "3xc": "Triple Captain" };

type Transfer = Manager["transfers"][number] & { gain: number | null };

export default function MyTeam() {
  const site = useSite();
  const manager = useData<Manager>("manager.json");
  const all = useAllGameweeks(site.meta.played);
  const gws = manager ? Object.keys(manager.gameweeks).map(Number).sort((a, b) => a - b) : [];
  const [chosen, setChosen] = useState<number | null>(null);
  const gw = chosen ?? gws[gws.length - 1];

  const hist = useMemo(() => (manager?.history ?? []).map((h) => ({
    ...h, average: site.meta.events.find((e) => e.id === h.event)?.average ?? null,
  })), [manager, site]);

  const pointsChart = useCallback((width: number) => Plot.plot({
    ...plotDefaults(width),
    height: 240,
    x: { label: null, type: "band", padding: barPadding(width, hist.length), tickFormat: (d: number) => `GW${d}` },
    y: { label: "Points", grid: true, nice: true },
    marks: [
      Plot.ruleY([0], { stroke: color.grid }),
      Plot.barY(hist, { x: "event", y: "points", fill: color.s1, ry2: 4 }),
      Plot.line(hist, { x: "event", y: "average", stroke: color.s2, strokeWidth: 2 }),
      Plot.dot(hist, { x: "event", y: "average", fill: color.s2, r: 4, stroke: color.surface, strokeWidth: 2 }),
      Plot.tip(hist, Plot.pointerX({ x: "event", y: "points",
        title: (h: typeof hist[number]) => `GW${h.event}: ${h.points} points (average ${h.average ?? "–"})` +
          (h.event_transfers_cost ? `\nincl. −${h.event_transfers_cost} for transfers` : "") })),
    ],
  }), [hist]);

  const rankChart = useCallback((width: number) => Plot.plot({
    ...plotDefaults(width),
    height: 240,
    marginLeft: 56,
    x: { label: null, type: "point", tickFormat: (d: number) => `GW${d}`, padding: 0.4 },
    y: { label: "Overall rank", reverse: true, grid: true, tickFormat: "~s", nice: true },
    marks: [
      Plot.line(hist, { x: "event", y: "overall_rank", stroke: color.s1, strokeWidth: 2 }),
      Plot.dot(hist, { x: "event", y: "overall_rank", fill: color.s1, r: 4, stroke: color.surface, strokeWidth: 2 }),
      Plot.tip(hist, Plot.pointerX({ x: "event", y: "overall_rank",
        title: (h: typeof hist[number]) => `GW${h.event}: rank ${h.overall_rank.toLocaleString()}` })),
    ],
  }), [hist]);

  const transfers: Transfer[] = useMemo(() => {
    if (!manager || !all) return [];
    const pointsFrom = (element: number, from: number) =>
      [...all.values()].filter((g) => g.gw >= from).reduce((s, g) => s + (totals(g).get(element)?.points ?? 0), 0);
    return manager.transfers.map((t) => ({ ...t, gain: pointsFrom(t.element_in, t.event) - pointsFrom(t.element_out, t.event) }));
  }, [manager, all]);

  if (manager === undefined) return <Loading />;
  if (manager === null) return <p>No team was exported. Run <code>xpfpl export --team-id &lt;id&gt;</code>.</p>;

  const latest = manager.history[manager.history.length - 1];
  const name = (id: number) => site.player.get(id)?.web_name ?? String(id);
  const week = gw ? manager.gameweeks[String(gw)] : undefined;
  const weekData = gw && all ? all.get(gw) : undefined;
  const weekTotals = weekData ? totals(weekData) : undefined;
  const h = manager.history.find((x) => x.event === gw);

  const transferColumns: Column<Transfer>[] = [
    { key: "gw", label: "GW", numeric: true, value: (t) => t.event },
    { key: "out", label: "Out", value: (t) => name(t.element_out) },
    { key: "sold", label: "Sold", numeric: true, value: (t) => t.element_out_cost, render: (t) => money(t.element_out_cost / 10) },
    { key: "in", label: "In", value: (t) => name(t.element_in) },
    { key: "bought", label: "Bought", numeric: true, value: (t) => t.element_in_cost, render: (t) => money(t.element_in_cost / 10) },
    { key: "gain", label: "Points since", numeric: true, value: (t) => t.gain,
      render: (t) => (t.gain === null ? "–" : <span className={t.gain >= 0 ? "good" : "bad"}>{signed(t.gain, 0)}</span>),
      title: "Points the player bought has scored since, minus the player sold" },
  ];

  return (
    <>
      <h2>{manager.name}</h2>
      <p className="lede">An FPL team's season: points, rank, each week's picks against the best possible from the same squad, and how the transfers worked out.</p>
      {latest && (
        <Tiles tiles={[
          { label: "Total points", value: int(latest.total_points) },
          { label: "Overall rank", value: compact(latest.overall_rank), note: latest.overall_rank.toLocaleString() },
          { label: "Team value", value: money(latest.value / 10), note: `${money(latest.bank / 10)} in the bank` },
          { label: "Points on the bench", value: int(manager.history.reduce((s, x) => s + x.points_on_bench, 0)) },
          { label: "Left on the table", note: "best XI and captain from the same 15, all season",
            value: int(Object.entries(manager.gameweeks).reduce((s, [g, w]) => {
              const hh = manager.history.find((x) => x.event === Number(g));
              return s + (hh ? w.best - (hh.points + hh.event_transfers_cost) : 0);
            }, 0)) },
        ]} />
      )}

      <div className="grid-2">
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Points per gameweek</h3>
          <Legend items={[{ label: "Points", color: color.s1 }, { label: "FPL average", color: color.s2, kind: "line" }]} />
          <Chart make={pointsChart} height={240} ariaLabel="Points per gameweek against the average" />
        </div>
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Overall rank</h3>
          <Chart make={rankChart} height={240} ariaLabel="Overall rank by gameweek" />
        </div>
      </div>

      <h3>Picks</h3>
      <div className="toolbar">
        <label>
          Gameweek
          <select value={gw} onChange={(e) => setChosen(Number(e.target.value))}>
            {[...gws].reverse().map((g) => <option key={g} value={g}>GW{g}</option>)}
          </select>
        </label>
        {week?.chip && <span className="tag">{CHIP_NAMES[week.chip] ?? week.chip}</span>}
      </div>
      {week && h && (
        <div className="card">
          <p style={{ marginTop: 0 }}>
            <strong>{h.points}</strong> points{h.event_transfers_cost ? ` (after −${h.event_transfers_cost} for transfers)` : ""}.
            With hindsight, the best XI and captain from the same 15 would have scored <strong>{week.best}</strong>
            {" "}({signed(week.best - (h.points + h.event_transfers_cost), 0)}).
            {(() => {
              const cap = week.picks.find((p) => p.captain);
              if (!cap || !weekTotals || cap.element === week.best_captain) return null;
              return <> Best captain: {name(week.best_captain)} ({weekTotals.get(week.best_captain)?.points ?? 0}) rather than {name(cap.element)} ({weekTotals.get(cap.element)?.points ?? 0}).</>;
            })()}
          </p>
          <div className="picks">
            {[1, 2, 3, 4].map((pos) => (
              <div className="picks-row" key={pos}>
                {week.picks.filter((p) => p.slot <= 11 && site.player.get(p.element)?.element_type === pos).map((p) => (
                  <Pick key={p.element} element={p.element} captain={p.captain} vice={p.vice} multiplier={p.multiplier}
                        inBest={week.best_xi.includes(p.element)} gw={weekTotals?.get(p.element)} />
                ))}
              </div>
            ))}
            <div className="picks-row" style={{ borderTop: "1px solid var(--grid)", paddingTop: 8 }}>
              {week.picks.filter((p) => p.slot > 11).map((p) => (
                <Pick key={p.element} element={p.element} bench multiplier={p.multiplier}
                      inBest={week.best_xi.includes(p.element)} gw={weekTotals?.get(p.element)} />
              ))}
            </div>
          </div>
          {week.auto_subs.length > 0 && (
            <p className="note">Auto-subs: {week.auto_subs.map((s) => `${name(s.element_in)} on for ${name(s.element_out)}`).join(", ")}.</p>
          )}
          <p className="note">Points count the captain's double. A dashed outline marks a player the hindsight-best XI would have started.</p>
        </div>
      )}

      <div className="grid-2">
        <div>
          <h3>Chips</h3>
          <table>
            <thead><tr><th>Chip</th><th>Window</th><th>Used</th></tr></thead>
            <tbody>
              {site.meta.chips.map((c) => {
                const used = manager.chips.find((u) => u.name === c.name && u.event >= c.start && u.event <= c.stop);
                return (
                  <tr key={`${c.name}-${c.start}`}>
                    <td>{CHIP_NAMES[c.name] ?? c.name}</td>
                    <td>GW{c.start}–{c.stop}</td>
                    <td>{used ? `GW${used.event}` : <span className="muted">Available</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div>
          <h3>Transfers</h3>
          {transfers.length ? <Table columns={transferColumns} data={transfers} sort="gw" rowKey={(t) => `${t.event}-${t.element_in}-${t.element_out}`} />
            : <p className="muted">No transfers yet.</p>}
        </div>
      </div>
      <Note>Picks are public on FPL once each deadline passes, so a chip or transfers for the next gameweek show only after its deadline.</Note>
    </>
  );
}

function Pick({ element, captain, vice, bench, multiplier, inBest, gw }: {
  element: number; captain?: boolean; vice?: boolean; bench?: boolean; multiplier: number; inBest: boolean;
  gw?: { points: number; xp: number | null; minutes: number };
}) {
  const site = useSite();
  const p = site.player.get(element);
  const counted = gw ? (bench ? gw.points : gw.points * multiplier) : null;
  return (
    <div className={`pick${bench ? " bench" : ""}`} style={inBest ? { outline: "1px dashed var(--s1)", outlineOffset: 1 } : undefined}
         title={gw ? `${gw.minutes} min · xP ${pts(gw.xp)}` : undefined}>
      <div className="badge">{captain ? "C" : vice ? "VC" : " "}</div>
      <div className="name">{p?.web_name ?? element}</div>
      <div><Club id={p?.team} /> <span className="muted">{p ? POSITIONS[p.element_type] : ""}</span></div>
      <div className="pts">{counted ?? "–"}</div>
      <div className="muted">xP {pts(gw?.xp)}</div>
    </div>
  );
}
