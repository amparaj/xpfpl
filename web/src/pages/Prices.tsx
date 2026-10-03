import * as Plot from "@observablehq/plot";
import { useCallback, useMemo, useState } from "react";
import { color } from "../colors";
import { Chart, Club, Legend, Loading, Note, Segmented, Sheet, Table, Tiles, plotDefaults, usePhone, type Column } from "../components/ui";
import type { ModelTeam } from "../data";
import { POSITIONS, dec, int, money, pts, signed, when } from "../format";
import { likelihood, pricesUrl, ukDay, type PriceChange, type PriceLog, type PricePlayer } from "../prices";
import { history, useAllGameweeks } from "../season";
import { useData, useSite } from "../site";

/** From this much progress either way, a player counts as close to a move. */
const NEAR = 70;

/** FPL's progress as a bar growing from the middle: right (green) towards a rise, left (red) towards a fall. */
function Progress({ value }: { value: number | null }) {
  if (value === null) return <span className="muted">–</span>;
  const v = Math.max(-100, Math.min(100, value));
  return (
    <span className="progress" title={`${dec(value, 1)} of ±100`}>
      <span className="progress-track">
        <span className={`progress-fill ${v >= 0 ? "up" : "down"}`}
              style={v >= 0 ? { left: "50%", width: `${v / 2}%` } : { right: "50%", width: `${-v / 2}%` }} />
      </span>
      <span className="progress-num">{signed(value, 1)}</span>
    </span>
  );
}

function PlayerPrice({ player, log, inSheet = false }: { player: PricePlayer; log: PriceLog; inSheet?: boolean }) {
  const site = useSite();
  const all = useAllGameweeks(site.meta.played);
  const byGw = useMemo(() => history(all, player.id), [all, player.id]);
  const progress = useMemo(() => log.times.map((t, i) => ({ t: new Date(t), v: player.h[i] }))
    .filter((d): d is { t: Date; v: number } => d.v !== null && d.v !== undefined), [log.times, player.h]);
  const moves = log.changes.filter((c) => c.id === player.id);

  const progressChart = useCallback((width: number) => Plot.plot({
    ...plotDefaults(width),
    height: 220,
    x: { label: "Your time", type: "time" },
    y: { label: "Progress", domain: [-100, 100], grid: true },
    marks: [
      Plot.ruleY([100], { stroke: color.good, strokeDasharray: "4,4" }),
      Plot.ruleY([-100], { stroke: color.bad, strokeDasharray: "4,4" }),
      Plot.ruleY([0], { stroke: color.grid }),
      Plot.line(progress, { x: "t", y: "v", stroke: color.s1, strokeWidth: 2 }),
      Plot.tip(progress, Plot.pointerX({ x: "t", y: "v", title: (d: { t: Date; v: number }) => `${when(d.t.toISOString())}: ${dec(d.v, 1)}` })),
    ],
  }), [progress]);

  const priceChart = useCallback((width: number) => Plot.plot({
    ...plotDefaults(width),
    height: 200,
    x: { label: "Gameweek", ticks: byGw.map((g) => g.gw), tickFormat: (d: number) => `GW${d}` },
    y: { label: "Price (£m)", grid: true, nice: true, tickFormat: (d: number) => d.toFixed(1) },
    marks: [
      Plot.line(byGw, { x: "gw", y: (g) => g.price, stroke: color.s1, strokeWidth: 2, curve: "step-after" }),
      Plot.dot(byGw, { x: "gw", y: (g) => g.price, fill: color.s1, r: 3 }),
      Plot.tip(byGw, Plot.pointerX({ x: "gw", y: (g) => g.price, title: (g) => `GW${g.gw}: ${money(g.price)}` })),
    ],
  }), [byGw]);

  return (
    <div className={inSheet ? undefined : "card"}>
      {!inSheet && (
        <div className="toolbar" style={{ justifyContent: "space-between" }}>
          <div>
            <strong style={{ fontSize: 17 }}>{player.name}</strong>{" "}
            <Club id={player.team} /> <span className="muted">{POSITIONS[player.pos]} · {money(player.cost / 10)}</span>
          </div>
          <a href="#prices" className="link">Close</a>
        </div>
      )}
      <Tiles tiles={[
        { label: "Progress now", value: player.pct === null ? "–" : signed(player.pct, 1), note: "+100 rises, −100 falls" },
        { label: "Per hour", value: player.rate === null ? "–" : signed(player.rate, 0) },
        { label: "Next update", value: likelihood(player.lik[0]) || "–", note: player.proj[0] !== null && player.proj[0] !== undefined ? `projected ${dec(player.proj[0], 1)}` : undefined },
        { label: "This season", value: signed(player.change_start / 10, 1), note: `${int(player.net_event)} net transfers this gameweek` },
      ]} />
      {progress.length > 1 ? (
        <>
          <h3>Progress, last three days</h3>
          <Chart make={progressChart} height={220} ariaLabel={`${player.name}: progress towards a price change over the last three days`} />
        </>
      ) : <Note>The progress chart fills in as the hourly log runs.</Note>}
      {all === undefined ? <Loading /> : byGw.length > 1 && (
        <>
          <h3>Price by gameweek</h3>
          <Chart make={priceChart} height={200} ariaLabel={`${player.name}: price by gameweek`} />
        </>
      )}
      {moves.length > 0 && (
        <p className="note">Moves logged: {moves.map((m, i) => (
          <span key={i}>{i > 0 && " · "}{m.t ? ukDay(m.t) : `GW${m.gw}`} {money(m.from / 10)} → {money(m.to / 10)}</span>))}
        </p>
      )}
    </div>
  );
}

type View = "rise" | "fall" | "all";
type Move = "all" | "rise" | "fall";

export default function Prices() {
  const site = useSite();
  const live = useData<PriceLog>(pricesUrl());
  const local = useData<PriceLog>(live === null ? "prices.json" : null);
  const log = live ?? local;
  const modelTeam = useData<ModelTeam>("modelteam.json");
  const [view, setView] = useState<View>("rise");
  const [pos, setPos] = useState(0);
  const [club, setClub] = useState(0);
  const [search, setSearch] = useState("");
  const [modelOnly, setModelOnly] = useState(false);
  const [move, setMove] = useState<Move>("all");
  const selectedId = Number(window.location.hash.split("/")[1]) || null;
  const phone = usePhone();

  const modelSquad = useMemo(() => {
    const weeks = modelTeam?.gameweeks ?? [];
    const latest = modelTeam?.next ?? weeks[weeks.length - 1];
    return new Set(latest?.squad ?? []);
  }, [modelTeam]);

  const list = useMemo(() => {
    if (!log) return [];
    const q = search.trim().toLowerCase();
    return log.players.filter((p) =>
      (view === "all" || (view === "rise" ? (p.pct ?? 0) > 0 : (p.pct ?? 0) < 0)) &&
      (!pos || p.pos === pos) && (!club || p.team === club) && (!modelOnly || modelSquad.has(p.id)) &&
      (!q || p.name.toLowerCase().includes(q)));
  }, [log, view, pos, club, search, modelOnly, modelSquad]);

  const changes = useMemo(() => (log?.changes ?? [])
    .filter((c) => move === "all" || (move === "rise" ? c.to > c.from : c.to < c.from))
    .map((c, i) => ({ ...c, key: i }))
    .sort((a, b) => (b.t ?? "").localeCompare(a.t ?? "") || (b.gw ?? 0) - (a.gw ?? 0)), [log, move]);

  if (log === undefined || (log === null && local === undefined)) return <Loading />;
  if (!log) return <p>The price tracker hasn't run yet.</p>;

  const selected = selectedId ? log.players.find((p) => p.id === selectedId) : undefined;
  const forecast = (id: number) => site.player.get(id)?.forecast ?? null;
  const columns: Column<PricePlayer>[] = [
    { key: "name", label: "Player", value: (p) => p.name, render: (p) => (
      <>{p.name}{modelSquad.has(p.id) && <> <span className="tag" title="In this model's team">MT</span></>}</>) },
    { key: "team", label: "Club", value: (p) => site.team.get(p.team)?.short, render: (p) => <Club id={p.team} /> },
    { key: "pos", label: "Pos", value: (p) => p.pos, render: (p) => POSITIONS[p.pos] },
    { key: "price", label: "£m", numeric: true, value: (p) => p.cost, render: (p) => dec(p.cost / 10, 1) },
    { key: "pct", label: "Progress", numeric: true, value: (p) => p.pct, render: (p) => <Progress value={p.pct} />,
      title: "FPL's progress towards this player's next price change: +100 rises, −100 falls" },
    { key: "rate", label: "Per hour", numeric: true, value: (p) => p.rate, render: (p) => (p.rate === null ? "–" : signed(p.rate, 0)),
      title: "How fast the progress is moving, per hour" },
    { key: "next", label: "Next update", value: (p) => p.lik[0] ?? null, render: (p) => likelihood(p.lik[0]) || "–",
      title: "FPL's likelihood of a move at the next price update (usually overnight UK time)" },
    { key: "after", label: "Update after", value: (p) => p.lik[1] ?? null, render: (p) => likelihood(p.lik[1]) || "–" },
    { key: "net", label: "Net transfers", numeric: true, value: (p) => p.net_event, render: (p) => signed(p.net_event, 0),
      title: "Transfers in minus out this gameweek" },
    { key: "sel", label: "Sel %", numeric: true, value: (p) => p.selected, render: (p) => dec(p.selected, 1) },
    { key: "gw", label: "This GW", numeric: true, value: (p) => p.change_event,
      render: (p) => (p.change_event ? signed(p.change_event / 10, 1) : "–"), title: "Price change this gameweek (£m)" },
    { key: "season", label: "Since GW1", numeric: true, value: (p) => p.change_start,
      render: (p) => (p.change_start ? signed(p.change_start / 10, 1) : "–"), title: "Price change this season (£m)" },
    { key: "xp", label: site.meta.next_gw ? `xP GW${site.meta.next_gw}` : "xP next", numeric: true, value: (p) => forecast(p.id),
      render: (p) => pts(forecast(p.id)), title: "This model's forecast for the next gameweek" },
  ];
  const changeColumns: Column<PriceChange & { key: number }>[] = [
    { key: "day", label: "Day (UK)", sortable: true, value: (c) => c.t ?? "", render: (c) => (c.t ? ukDay(c.t) : `GW${c.gw ?? "?"}`) },
    { key: "name", label: "Player", value: (c) => c.name },
    { key: "team", label: "Club", value: (c) => site.teamByCode.get(c.team_code)?.short, render: (c) => <Club code={c.team_code} /> },
    { key: "pos", label: "Pos", value: (c) => c.pos, render: (c) => POSITIONS[c.pos] },
    { key: "move", label: "Move", value: (c) => c.to - c.from,
      render: (c) => <span className={c.to > c.from ? "good" : "bad"}>{c.to > c.from ? "▲ Rise" : "▼ Fall"}</span> },
    { key: "from", label: "From", numeric: true, value: (c) => c.from, render: (c) => money(c.from / 10) },
    { key: "to", label: "To", numeric: true, value: (c) => c.to, render: (c) => money(c.to / 10) },
  ];

  const rises = log.players.filter((p) => p.change_event > 0).length;
  const falls = log.players.filter((p) => p.change_event < 0).length;
  const sinceStart = log.changes.filter((c) => c.t).length;
  return (
    <>
      <h2>Prices</h2>
      <p className="lede">Who has just risen or fallen in price, and who's closest to the next move. Progress is FPL's own
        figure: a player rises when it reaches +100 and falls at −100, at one of FPL's price updates (usually overnight
        UK time). It's refreshed every hour; click a player for the last three days and his price this season.</p>
      <Tiles tiles={[
        { label: `Rises in GW${log.gw_current ?? "?"}`, value: rises },
        { label: `Falls in GW${log.gw_current ?? "?"}`, value: falls },
        { label: "Close to a rise", value: log.players.filter((p) => (p.pct ?? 0) >= NEAR).length, note: `progress +${NEAR} or more` },
        { label: "Close to a fall", value: log.players.filter((p) => (p.pct ?? 0) <= -NEAR).length, note: `progress −${NEAR} or less` },
        { label: "Last checked", value: <span style={{ fontSize: 17 }}>{when(log.updated)}</span>, note: live ? "every hour" : "when the site was published" },
      ]} />
      {selected && !phone && <PlayerPrice player={selected} log={log} />}
      {phone && (
        <Sheet open={!!selected} onClose={() => { window.location.hash = "prices"; }}
               title={selected && <>{selected.name} <Club id={selected.team} /> <span className="muted">{POSITIONS[selected.pos]} · {money(selected.cost / 10)}</span></>}>
          {selected && <PlayerPrice player={selected} log={log} inSheet />}
        </Sheet>
      )}
      <div className="toolbar" style={{ marginTop: 14 }}>
        <Segmented label="Direction" value={view} onChange={setView}
                   options={[{ value: "rise", label: "Towards a rise" }, { value: "fall", label: "Towards a fall" }, { value: "all", label: "All" }]} />
        <Segmented label="Position" value={pos} onChange={setPos}
                   options={[{ value: 0, label: "All" }, ...Object.entries(POSITIONS).map(([k, v]) => ({ value: Number(k), label: v }))]} />
        <select aria-label="Club" value={club} onChange={(e) => setClub(Number(e.target.value))}>
          <option value={0}>All clubs</option>
          {[...site.meta.teams].sort((a, b) => a.name.localeCompare(b.name)).map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
        <input type="search" placeholder="Search name" aria-label="Search players" value={search} onChange={(e) => setSearch(e.target.value)} />
        <label><input type="checkbox" checked={modelOnly} onChange={(e) => setModelOnly(e.target.checked)} /> This model's team only</label>
      </div>
      <Table key={view} columns={columns} data={list} sort="pct" desc={view !== "fall"} rowKey={(p) => p.id} limit={40}
             selected={selectedId} cardSub={["team", "pos", "price"]} cardStats={["pct", "next"]}
             onRow={(p) => { window.location.hash = `prices/${p.id}`; if (!phone) window.scrollTo({ top: 0, behavior: "smooth" }); }} />
      <Legend items={[{ label: "Towards a rise", color: color.good }, { label: "Towards a fall", color: color.bad }]} />

      <h3>Price changes</h3>
      <div className="toolbar">
        <Segmented label="Moves" value={move} onChange={setMove}
                   options={[{ value: "all", label: "All" }, { value: "rise", label: "Rises" }, { value: "fall", label: "Falls" }]} />
        <span className="muted">{changes.length} moves</span>
      </div>
      <Table columns={changeColumns} data={changes} rowKey={(c) => c.key} limit={30} cardSub={["team", "pos", "day"]} cardStats={["move", "to"]} />
      <Note>
        A move's day is the hourly check that first saw it{sinceStart < log.changes.length && <>; moves shown by gameweek
        were made before the log started</>}. Prices are FPL's own; this model values a likely rise a little when it
        picks a team (a riser bought now is worth more later), estimated from form and the week's transfers. FPL's progress
        figures are new this season and are being kept day by day, so this model can learn from them once there's
        enough history.
      </Note>
    </>
  );
}
