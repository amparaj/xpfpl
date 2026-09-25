import * as Plot from "@observablehq/plot";
import { useCallback, useMemo, useState } from "react";
import { color } from "../colors";
import { Chart, barPadding, Club, Legend, Loading, Note, Table, Tiles, plotDefaults, type Column } from "../components/ui";
import type { ModelTeam, ModelWeek } from "../data";
import { POSITIONS, int, money, pts, signed, when } from "../format";
import { totals, useAllGameweeks, type PlayerGw } from "../season";
import { useData, useSite } from "../site";

const CHIP_NAMES: Record<string, string> = { wildcard: "Wildcard", freehit: "Free Hit", bboost: "Bench Boost", "3xc": "Triple Captain" };
const SOURCE: Record<ModelWeek["source"], string> = {
  live: "Live", replay: "Replay", carried: "Carried over",
};

type Week = ModelWeek & { average: number | null; highest: number | null };

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
      Plot.tip(weeks, Plot.pointerX({ x: "gw", y: "points",
        title: (w: Week) => `GW${w.gw} (${SOURCE[w.source].toLowerCase()}): ${w.points} points, FPL average ${w.average ?? "–"}` +
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

  const week = gw === data.next?.gw ? data.next : weeks.find((w) => w.gw === gw);
  const played = week && week.points !== undefined;
  const scores = played && all?.get(week.gw) ? totals(all.get(week.gw)!) : undefined;

  const columns: Column<Week>[] = [
    { key: "gw", label: "GW", numeric: true, value: (w) => w.gw },
    { key: "source", label: "Decision", value: (w) => w.source, render: (w) => <span className={w.source === "live" ? "tag" : "muted"}>{SOURCE[w.source]}</span>,
      title: "Live: saved before the deadline. Replay: filled in by the backtest for the weeks before the live record began." },
    { key: "points", label: "Points", numeric: true, value: (w) => w.points, render: (w) => <strong>{w.points}</strong> },
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
        A paper FPL team that does exactly what the model says. Before every deadline it makes its transfers, picks its
        XI, bench order, captain and chips with the same optimiser and settings as the app, and the decision is saved
        before the deadline. Then it's scored like any FPL team: auto-subs, the vice-captain, chips and transfer hits.
      </p>
      {weeks.length > 0 && (
        <Tiles tiles={[
          { label: "Total points", value: int(total), note: `${weeks.length} gameweek${weeks.length === 1 ? "" : "s"}` },
          { label: "Against the FPL average", value: signed(total - average, 0), note: `above average in ${above} of ${weeks.length}` },
          { label: "Transfers", value: int(moves), note: hits ? `${hits} hit${hits === 1 ? "" : "s"} (−${4 * hits})` : "no hits taken" },
          { label: "Left on the table", value: int(missed), note: "best XI and captain from the same 15, all season" },
        ]} />
      )}

      {weeks.length > 0 && (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Points per gameweek</h3>
          <Legend items={[{ label: "Live", color: color.s1 }, ...(replayed.length ? [{ label: "Replay", color: `color-mix(in srgb, ${color.s1} 45%, transparent)` }] : []),
                          { label: "FPL average", color: color.s2, kind: "line" as const }]} />
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
              {" "}With hindsight, the best XI and captain from the same 15 would have scored <strong>{int(week.best)}</strong>.</>
              : <>Expected <strong>{pts(week.xp)}</strong> points (xP, captain doubled). This can still change until the deadline, {when(site.meta.next_deadline)}.</>}
            {" "}{week.transfers.length ? `Transfers: ${week.transfers.map((t) => `${name(t.out)} (${money(t.sold)}) → ${name(t.in)} (${money(t.bought)})`).join(", ")}.`
              : week.source === "carried" ? "No decision was saved this week, so last week's team played on." : "No transfers."}
            {" "}{money(week.bank)} in the bank and {week.free_transfers} free transfer{week.free_transfers === 1 ? "" : "s"} before the week.
          </p>
          <div className="picks">
            {[1, 2, 3, 4].map((pos) => (
              <div className="picks-row" key={pos}>
                {week.lineup.filter((p) => site.player.get(p)?.element_type === pos).map((p) => (
                  <Pick key={p} element={p} captain={p === week.captain} vice={p === week.vice}
                        multiplier={p === week.captain_played || (!played && p === week.captain) ? (week.chip === "3xc" ? 3 : 2) : 1}
                        inBest={!!week.best_xi?.includes(p)} gw={scores?.get(p)} played={!!played} />
                ))}
              </div>
            ))}
            <div className="picks-row" style={{ borderTop: "1px solid var(--grid)", paddingTop: 8 }}>
              {week.bench.map((p) => (
                <Pick key={p} element={p} bench multiplier={week.chip === "bboost" ? 1 : 0}
                      inBest={!!week.best_xi?.includes(p)} gw={scores?.get(p)} played={!!played} />
              ))}
            </div>
          </div>
          {!!week.autosubs?.length && (
            <p className="note">Auto-subs: {week.autosubs.map(([out, sub]) => `${name(sub)} on for ${name(out)}`).join(", ")}.</p>
          )}
          <p className="note">
            {played ? "Points count the captain's multiplier; bench points count only with Bench Boost. A dashed outline marks a player the hindsight-best XI would have started."
              : `Each player's xP is the model's forecast for GW${week.gw}.`}
          </p>
        </div>
      )}

      {weeks.length > 0 && (
        <>
          <h3>Week by week</h3>
          <Table columns={columns} data={[...weeks].reverse()} rowKey={(w) => w.gw} />
          <dl className="defs">
            <dt>Live</dt>
            <dd>
              Decided and saved before that gameweek's deadline, using only what was known then: the honest record, like
              entering a real team.
            </dd>
            <dt>Replay</dt>
            <dd>
              Reconstructed afterwards, for the weeks before the live record began. The backtest (the same tool that tests
              the model on past seasons) decided them with a model trained only on earlier seasons, each player's form as
              it stood at each deadline, and the same optimiser, settings and chip rules. Two things make a replay a little
              less trustworthy: it has no injury news, so a player who had been ruled out can look available, and it was
              decided after the matches, so it relies on the code being fair rather than a timestamp before the deadline.
              That's why the chart shows replays paler.
            </dd>
            <dt>Carried over</dt>
            <dd>No decision was saved before the deadline, so last week's team played on and a free transfer was banked, as FPL does.</dd>
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

function Pick({ element, captain, vice, bench, multiplier, inBest, gw, played }: {
  element: number; captain?: boolean; vice?: boolean; bench?: boolean; multiplier: number; inBest: boolean;
  gw?: PlayerGw; played: boolean;
}) {
  const site = useSite();
  const p = site.player.get(element);
  const shown = played ? (gw ? gw.points * Math.max(multiplier, bench ? 1 : 0) : 0) : null;
  return (
    <div className={`pick${bench ? " bench" : ""}`} style={inBest ? { outline: "1px dashed var(--s1)", outlineOffset: 1 } : undefined}
         title={gw ? `${gw.minutes} min · xP ${pts(gw.xp)}` : undefined}>
      <div className="badge">{captain ? "C" : vice ? "VC" : " "}</div>
      <div className="name">{p?.web_name ?? element}</div>
      <div><Club id={p?.team} /> <span className="muted">{p ? POSITIONS[p.element_type] : ""}</span></div>
      {played ? <div className="pts">{shown}</div> : <div className="pts">{pts(p?.forecast)}</div>}
      <div className="muted">{played ? `xP ${pts(gw?.xp)}` : "xP"}</div>
    </div>
  );
}
