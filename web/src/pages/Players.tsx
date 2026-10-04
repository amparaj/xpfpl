import * as Plot from "@observablehq/plot";
import { useCallback, useEffect, useMemo, useState } from "react";
import { color } from "../colors";
import { Profile } from "../components/Profile";
import { NewsDetail, NewsStatus, PressTag, useNewsLog } from "../components/PlayerNews";
import { Chart, barPadding, Club, Legend, Loading, Note, Segmented, Sheet, Table, plotDefaults, usePhone, type Column } from "../components/ui";
import { WhyProjection } from "../components/Why";
import { rows, type Forecast, type NextGw, type Player } from "../data";
import { POSITIONS, STATUS, dec, flagLabel, money, pct, pts, signed, when } from "../format";
import { ago, latestHeadlines, shortDay, type NewsChange, type NewsLog, type NewsPlayer } from "../news";
import { history, useAllGameweeks, type PlayerGw } from "../season";
import { useData, useHash, useSite } from "../site";
import { SquadFilter, SquadTags, inSquad, useSquads, type Squad } from "../squads";

type Status = "all" | "out" | "doubt";
type Columns = "stats" | "news";
type Since = 0 | 2 | 7;
const OUT = ["i", "s", "n"];

/** The forecast saved for the next gameweek, his row of it: for "Why this projection?". */
function useForecastRow(id: number) {
  const next = useData<NextGw>("next.json");
  return useMemo(() => {
    const row = rows<Forecast>(next?.players).find((f) => f.element === id);
    return next && row ? { row, gw: next.gw, gameweeks: next.gameweeks } : null;
  }, [next, id]);
}

function PlayerDetail({ player, inSheet = false, log }: { player: Player; inSheet?: boolean; log: NewsLog | null | undefined }) {
  const site = useSite();
  const forecast = useForecastRow(player.id);
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
  const why = (forecast || player.forecast !== null) && site.meta.next_gw && (
    <details className="why-details" open={inSheet}>
      <summary>Why this projection? <span className="muted">GW{forecast?.gw ?? site.meta.next_gw}: {pts(forecast ? forecast.row[`xp_${forecast.gw}`] : player.forecast)} xP</span></summary>
      <WhyProjection player={player} forecast={forecast?.row} gw={forecast?.gw ?? site.meta.next_gw} gameweeks={forecast?.gameweeks} />
    </details>
  );
  return (
    <div className={inSheet ? undefined : "card"}>
      {!inSheet && (
        <div className="toolbar" style={{ justifyContent: "space-between" }}>
          <div>
            <strong style={{ fontSize: 17 }}>{player.first_name} {player.second_name}</strong>{" "}
            <Club id={player.team} /> <span className="muted">{POSITIONS[player.element_type]} · {money(player.now_cost)}</span>
          </div>
          <a href="#players" className="link">Close</a>
        </div>
      )}
      {log ? <NewsDetail id={player.id} p={log.players.find((n) => n.id === player.id)} log={log} forecast={forecast?.row} />
        : (status || player.news) && (
          <p className="note" style={{ marginTop: 0 }}>
            <span className="tag warn">{status ?? "News"}</span> {player.news}
          </p>
        )}
      {inSheet && why}
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
      <Profile player={player} />
      {!inSheet && why}
    </div>
  );
}

export default function Players() {
  const site = useSite();
  const [pos, setPos] = useState(0);
  const [club, setClub] = useState(0);
  const [search, setSearch] = useState("");
  const [playedOnly, setPlayedOnly] = useState(true);
  // #players/out and #players/doubtful (Team News links there) open the news view.
  const opening = window.location.hash.split("/")[1];
  const [status, setStatusRaw] = useState<Status>(opening === "out" ? "out" : opening === "doubtful" ? "doubt" : "all");
  const [cols, setCols] = useState<Columns>(status === "all" ? "stats" : "news");
  const [since, setSince] = useState<Since>(0);
  const [squad, setSquad] = useState<Squad>("all");
  const setStatus = (v: Status) => { setStatusRaw(v); if (v !== "all") setCols("news"); };
  const hash = useHash();
  useEffect(() => {
    const view = hash.split("/")[1];
    if (view === "out" || view === "doubtful") setStatus(view === "out" ? "out" : "doubt");
  }, [hash]);
  const selectedId = Number(window.location.hash.split("/")[1]) || null;
  const selected = selectedId ? site.player.get(selectedId) : undefined;
  const phone = usePhone();
  const log = useNewsLog();
  const squads = useSquads();
  const nextData = useData<NextGw>("next.json");
  const forecasts = useMemo(() => new Map(rows<Forecast>(nextData?.players).map((f) => [f.element, f])), [nextData]);

  // FPL's news as of the last hourly check where there is one, else as at the export.
  const newsOf = useMemo(() => new Map((log?.players ?? []).map((n) => [n.id, n])), [log]);
  const news = (p: Player): NewsPlayer | undefined => newsOf.get(p.id);
  const statusOf = (p: Player) => (log ? news(p)?.status ?? "a" : p.status);
  const chanceOf = (p: Player) => (log ? news(p)?.chance ?? null : p.chance_of_playing_next_round);
  const headlineOf = useMemo(() => latestHeadlines(log), [log]);

  const list = useMemo(() => {
    const q = search.trim().toLowerCase();
    const now = Date.now();
    return site.players.filter((p) => {
      const s = statusOf(p), n = news(p);
      return (status === "all" ? (!playedOnly || p.minutes > 0) : status === "out" ? OUT.includes(s) : s === "d") &&
        (cols !== "news" || !since || (n?.added != null && now - Date.parse(n.added) <= since * 86_400_000)) &&
        (!pos || p.element_type === pos) && (!club || p.team === club) && inSquad(squads, squad, p.id) &&
        (!q || `${p.web_name} ${p.first_name} ${p.second_name}`.toLowerCase().includes(q));
    });
  }, [site.players, pos, club, search, playedOnly, status, cols, since, squad, squads, newsOf, log]); // eslint-disable-line react-hooks/exhaustive-deps

  const changes = useMemo(() => {
    const cutoff = Date.now() - 14 * 86_400_000;
    return (log?.log ?? []).filter((c) => c.t && Date.parse(c.t) >= cutoff)
      .map((c, i) => ({ ...c, key: i })).sort((a, b) => (b.t ?? "").localeCompare(a.t ?? ""));
  }, [log]);

  const next = site.meta.next_gw;
  const nameColumn: Column<Player> = { key: "name", label: "Player", value: (p) => p.web_name, render: (p) => (
    <>{p.web_name}{cols === "stats" && statusOf(p) !== "a" && <> <span className="tag warn" title={news(p)?.news ?? p.news}>
      {statusOf(p) === "d" && chanceOf(p) != null ? `${chanceOf(p)}%` : STATUS[statusOf(p)] ?? "Out"}</span></>}
      <SquadTags id={p.id} squads={squads} /></>) };
  const clubColumns: Column<Player>[] = [
    { key: "team", label: "Club", value: (p) => site.team.get(p.team)?.short, render: (p) => <Club id={p.team} /> },
    { key: "pos", label: "Pos", value: (p) => p.element_type, render: (p) => POSITIONS[p.element_type] },
  ];
  const forecastColumn: Column<Player> = { key: "forecast", label: next ? `xP GW${next}` : "xP next", short: "Next xP", numeric: true,
    value: (p) => p.forecast, render: (p) => pts(p.forecast), title: "This model's forecast for the next gameweek, saved before its deadline" };
  const newsColumns: Column<Player>[] = [
    nameColumn, ...clubColumns,
    { key: "status", label: "Status", sortable: true, value: (p) => chanceOf(p) ?? (statusOf(p) === "a" ? 100 : 0),
      render: (p) => <NewsStatus status={statusOf(p)} chance={chanceOf(p)} />, title: "FPL's status and chance of playing next round" },
    { key: "reason", label: "Reason", value: (p) => news(p)?.reason ?? "", wrap: true },
    { key: "back", label: "Back", sortable: true, value: (p) => news(p)?.back ?? "", render: (p) => shortDay(news(p)?.back),
      title: "FPL's expected return date" },
    { key: "press", label: "Press", value: (p) => news(p)?.press ?? "", render: (p) => <PressTag said={news(p)?.press ?? null} />,
      title: "What the manager said at his press conference before this gameweek" },
    { key: "used", label: "Chance used", numeric: true, value: (p) => forecasts.get(p.id)?.avail ?? null,
      render: (p) => pct(forecasts.get(p.id)?.avail), title: "The chance of playing this model used for the next gameweek" },
    { key: "updated", label: "Updated", sortable: true, value: (p) => news(p)?.added ?? "", render: (p) => (news(p)?.added ? ago(news(p)!.added) : "–"),
      title: "When FPL last changed this player's news" },
    forecastColumn,
    { key: "sel", label: "Sel %", numeric: true, value: (p) => p.selected_by_percent, render: (p) => dec(p.selected_by_percent, 1) },
    { key: "source", label: "Source", value: (p) => news(p)?.source ?? headlineOf.get(p.id)?.url ?? "",
      title: "FPL's source article, else the latest headline naming him (NewsNow)",
      render: (p) => {
        const article = news(p)?.source, line = headlineOf.get(p.id);
        if (article) return <a className="link" href={article} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>Article ↗</a>;
        if (line) return <a className="link" href={line.url} target="_blank" rel="noreferrer nofollow" title={`${line.title}${line.publisher ? ` (${line.publisher})` : ""}`}
                            onClick={(e) => e.stopPropagation()}>Headline ↗</a>;
        return "–";
      } },
  ];
  const changeColumns: Column<NewsChange & { key: number }>[] = [
    { key: "t", label: "When", sortable: true, value: (c) => c.t ?? "", render: (c) => when(c.t) },
    { key: "name", label: "Player", value: (c) => c.name, render: (c) => <a href={`#players/${c.id}`}>{c.name}</a> },
    { key: "team", label: "Club", value: (c) => site.teamByCode.get(c.team_code)?.short, render: (c) => <Club code={c.team_code} /> },
    { key: "status", label: "Status", value: (c) => STATUS[c.status] ?? "Available",
      render: (c) => <span className={c.status === "a" ? "good" : c.status === "d" ? undefined : "bad"}>{flagLabel(c.status, c.chance)}</span> },
    { key: "news", label: "News", value: (c) => c.news || "Back to full fitness", wrap: true },
  ];
  const columns: Column<Player>[] = [
    nameColumn, ...clubColumns,
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
    forecastColumn,
  ];

  return (
    <>
      <h2>Players</h2>
      <p className="lede">Every player's season so far and his team news. Click a player for his news (FPL's, the press
        conference's and the headlines), his points against xP week by week, where he plays and shoots from, and the players
        with the most similar profile. <a href="#news">Team News</a> has the same news club by club.</p>
      {selected && !phone && <PlayerDetail player={selected} log={log} />}
      {phone && (
        <Sheet open={!!selected} onClose={() => { window.location.hash = "players"; }}
               title={selected && <>{selected.web_name} <Club id={selected.team} /> <span className="muted">{POSITIONS[selected.element_type]} · {money(selected.now_cost)}</span></>}>
          {selected && <PlayerDetail player={selected} log={log} inSheet />}
        </Sheet>
      )}
      <div className="toolbar" style={{ marginTop: 14 }}>
        <Segmented label="Status" value={status} onChange={setStatus}
                   options={[{ value: "all", label: "All" }, { value: "out", label: "Out" }, { value: "doubt", label: "Doubtful" }]} />
        <Segmented label="Columns" value={cols} onChange={setCols}
                   options={[{ value: "stats", label: "Stats" }, { value: "news", label: "News" }]} />
        {cols === "news" && (
          <Segmented label="News changed" value={since} onChange={setSince}
                     options={[{ value: 0, label: "Any time" }, { value: 2, label: "48 h" }, { value: 7, label: "7 days" }]} />
        )}
      </div>
      <div className="toolbar">
        <SquadFilter value={squad} onChange={setSquad} squads={squads} />
      </div>
      <div className="toolbar">
        <Segmented label="Position" value={pos} onChange={setPos}
                   options={[{ value: 0, label: "All" }, ...Object.entries(POSITIONS).map(([k, v]) => ({ value: Number(k), label: v }))]} />
        <select aria-label="Club" value={club} onChange={(e) => setClub(Number(e.target.value))}>
          <option value={0}>All clubs</option>
          {[...site.meta.teams].sort((a, b) => a.name.localeCompare(b.name)).map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
        <input type="search" placeholder="Search name" aria-label="Search players" value={search} onChange={(e) => setSearch(e.target.value)} />
        {status === "all" && <label><input type="checkbox" checked={playedOnly} onChange={(e) => setPlayedOnly(e.target.checked)} /> Played this season</label>}
        <span className="muted">{list.length} players</span>
      </div>
      {cols === "stats" ? (
        <Table key="stats" columns={columns} data={list} sort="points" rowKey={(p) => p.id} limit={60} selected={selectedId}
               cardSub={["team", "pos"]} cardStats={["points", "forecast"]}
               onRow={(p) => { window.location.hash = `players/${p.id}`; if (!phone) window.scrollTo({ top: 0, behavior: "smooth" }); }} />
      ) : (
        <Table key="news" columns={newsColumns} data={list} sort="updated" rowKey={(p) => p.id} limit={60} selected={selectedId}
               cardSub={["team", "pos", "reason"]} cardStats={["status", "back"]}
               onRow={(p) => { window.location.hash = `players/${p.id}`; if (!phone) window.scrollTo({ top: 0, behavior: "smooth" }); }} />
      )}
      <Note>
        Prices, ownership and form are as FPL showed them when the site was last updated; the news is checked every hour.
        The red tag is FPL's flag (a doubt shows its chance of playing next round); "Mine" marks My Team and "MT" this model's team. "Back" is FPL's
        expected return date: he should be available for matches from that day. Source is the article FPL's news came from,
        or failing that the latest headline naming him.
      </Note>
      {cols === "news" && log && (
        <>
          <h3>Changes to FPL's news, last 14 days</h3>
          <Table columns={changeColumns} data={changes} sort="t" desc rowKey={(c) => c.key} limit={30}
                 cardSub={["team", "status"]} cardStats={["t"]} />
        </>
      )}
    </>
  );
}
