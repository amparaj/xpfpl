import * as Plot from "@observablehq/plot";
import { useCallback, useMemo, useState } from "react";
import { color } from "../colors";
import { Chart, barPadding, Club, Legend, Loading, Note, Segmented, Table, plotDefaults, type Column } from "../components/ui";
import type { Player } from "../data";
import { POSITIONS, dec, money, pts, signed } from "../format";
import { history, useAllGameweeks, type PlayerGw } from "../season";
import { useSite } from "../site";

const STATUS: Record<string, string> = { d: "Doubtful", i: "Injured", s: "Suspended", u: "Unavailable", n: "Not in squad" };

function PlayerDetail({ player }: { player: Player }) {
  const site = useSite();
  const all = useAllGameweeks(site.meta.played);
  const games = useMemo(() => history(all, player.id), [all, player.id]);
  const xpTotal = games.reduce((s, g) => s + (g.xp ?? 0), 0);
  const ptsTotal = games.reduce((s, g) => s + g.points, 0);

  const chart = useCallback((width: number) => Plot.plot({
    ...plotDefaults(width),
    height: 240,
    x: { label: "Gameweek", tickFormat: (d: number) => `GW${d}`, type: "band", padding: barPadding(width, games.length) },
    y: { label: "Points", grid: true, nice: true },
    marks: [
      Plot.ruleY([0], { stroke: color.grid }),
      Plot.barY(games, { x: "gw", y: "points", fill: color.s1, ry2: 4, insetLeft: 1, insetRight: 1 }),
      Plot.line(games.filter((g) => g.xp !== null), { x: "gw", y: "xp", stroke: color.s2, strokeWidth: 2 }),
      Plot.dot(games.filter((g) => g.xp !== null), { x: "gw", y: "xp", fill: color.s2, r: 4, stroke: color.surface, strokeWidth: 2 }),
      Plot.tip(games, Plot.pointerX({
        x: "gw", y: "points",
        title: (g: PlayerGw) => `GW${g.gw}: ${g.points} points · xP ${pts(g.xp)}\n${g.minutes} min · ${g.goals} G · ${g.assists} A · ${g.bonus} bonus`,
      })),
    ],
  }), [games]);

  const status = STATUS[player.status];
  return (
    <div className="card">
      <div className="toolbar" style={{ justifyContent: "space-between" }}>
        <div>
          <strong style={{ fontSize: 17 }}>{player.first_name} {player.second_name}</strong>{" "}
          <Club id={player.team} /> <span className="muted">{POSITIONS[player.element_type]} · {money(player.now_cost)}</span>
        </div>
        <a href="#players" className="link">Close</a>
      </div>
      {(status || player.news) && (
        <p className="note" style={{ marginTop: 0 }}>
          <span className="tag warn">{status ?? "News"}</span> {player.news}
        </p>
      )}
      {all === undefined ? <Loading /> : (
        <>
          <Legend items={[{ label: "Points", color: color.s1 }, { label: "xP before the deadline", color: color.s2, kind: "line" }]} />
          <Chart make={chart} height={240} ariaLabel={`${player.web_name}: points and xP by gameweek`} />
          <p className="note">
            {ptsTotal} points against {pts(xpTotal)} xP over {games.length} gameweeks ({signed(ptsTotal - xpTotal)}).
            {player.forecast !== null && <> Forecast for GW{site.meta.next_gw}: <strong>{pts(player.forecast)}</strong> xP.</>}
          </p>
        </>
      )}
    </div>
  );
}

export default function Players() {
  const site = useSite();
  const [pos, setPos] = useState(0);
  const [club, setClub] = useState(0);
  const [search, setSearch] = useState("");
  const [playedOnly, setPlayedOnly] = useState(true);
  const selectedId = Number(window.location.hash.split("/")[1]) || null;
  const selected = selectedId ? site.player.get(selectedId) : undefined;

  const list = useMemo(() => {
    const q = search.trim().toLowerCase();
    return site.players.filter((p) =>
      (!pos || p.element_type === pos) && (!club || p.team === club) && (!playedOnly || p.minutes > 0) &&
      (!q || `${p.web_name} ${p.first_name} ${p.second_name}`.toLowerCase().includes(q)));
  }, [site.players, pos, club, search, playedOnly]);

  const next = site.meta.next_gw;
  const columns: Column<Player>[] = [
    { key: "name", label: "Player", value: (p) => p.web_name, render: (p) => (
      <>{p.web_name}{p.status !== "a" && <> <span className="tag warn" title={p.news}>{p.chance_of_playing_next_round ?? 0}%</span></>}</>) },
    { key: "team", label: "Club", value: (p) => site.team.get(p.team)?.short, render: (p) => <Club id={p.team} /> },
    { key: "pos", label: "Pos", value: (p) => p.element_type, render: (p) => POSITIONS[p.element_type] },
    { key: "price", label: "£m", numeric: true, value: (p) => p.now_cost, render: (p) => dec(p.now_cost, 1) },
    { key: "rise", label: "Since GW1", numeric: true, value: (p) => p.cost_change_start / 10,
      render: (p) => (p.cost_change_start ? signed(p.cost_change_start / 10, 1) : "–"), title: "Price change this season (£m)" },
    { key: "sel", label: "Sel %", numeric: true, value: (p) => p.selected_by_percent, render: (p) => dec(p.selected_by_percent, 1) },
    { key: "points", label: "Points", numeric: true, value: (p) => p.total_points },
    { key: "minutes", label: "Mins", numeric: true, value: (p) => p.minutes },
    { key: "goals", label: "G", numeric: true, value: (p) => p.goals_scored, title: "Goals" },
    { key: "assists", label: "A", numeric: true, value: (p) => p.assists, title: "Assists" },
    { key: "cs", label: "CS", numeric: true, value: (p) => p.clean_sheets, title: "Clean sheets" },
    { key: "bonus", label: "Bonus", numeric: true, value: (p) => p.bonus },
    { key: "xg", label: "xG", numeric: true, value: (p) => p.expected_goals, render: (p) => dec(p.expected_goals) },
    { key: "xa", label: "xA", numeric: true, value: (p) => p.expected_assists, render: (p) => dec(p.expected_assists) },
    { key: "dc", label: "DC", numeric: true, value: (p) => p.defensive_contribution, title: "Defensive contribution" },
    { key: "form", label: "Form", numeric: true, value: (p) => p.form, render: (p) => dec(p.form, 1), title: "FPL's form: points per match over the last 30 days" },
    { key: "forecast", label: next ? `xP GW${next}` : "xP next", numeric: true, value: (p) => p.forecast,
      render: (p) => pts(p.forecast), title: "The model's forecast for the next gameweek, saved before its deadline" },
  ];

  return (
    <>
      <h2>Players</h2>
      <p className="lede">Every player's season so far. Click a player to see their points against xP week by week.</p>
      {selected && <PlayerDetail player={selected} />}
      <div className="toolbar" style={{ marginTop: 14 }}>
        <Segmented label="Position" value={pos} onChange={setPos}
                   options={[{ value: 0, label: "All" }, ...Object.entries(POSITIONS).map(([k, v]) => ({ value: Number(k), label: v }))]} />
        <select aria-label="Club" value={club} onChange={(e) => setClub(Number(e.target.value))}>
          <option value={0}>All clubs</option>
          {[...site.meta.teams].sort((a, b) => a.name.localeCompare(b.name)).map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
        <input type="search" placeholder="Search name" aria-label="Search players" value={search} onChange={(e) => setSearch(e.target.value)} />
        <label><input type="checkbox" checked={playedOnly} onChange={(e) => setPlayedOnly(e.target.checked)} /> Played this season</label>
        <span className="muted">{list.length} players</span>
      </div>
      <Table columns={columns} data={list} sort="points" rowKey={(p) => p.id} limit={60} selected={selectedId}
             onRow={(p) => { window.location.hash = `players/${p.id}`; window.scrollTo({ top: 0, behavior: "smooth" }); }} />
      <Note>Prices, ownership, form and news are as FPL showed them when the site was last updated. A red % tag is FPL's chance of playing next round.</Note>
    </>
  );
}
