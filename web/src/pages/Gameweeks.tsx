import * as Plot from "@observablehq/plot";
import { useCallback, useMemo, useState } from "react";
import { color } from "../colors";
import { Chart, Club, Note, Loading, Table, Tiles, plotDefaults, type Column } from "../components/ui";
import { competition, side } from "../midweek";
import { gwFile, rows, type Gameweek, type GwRow } from "../data";
import { POSITIONS, compact, dec, int, pts, signed, when } from "../format";
import { totals, type PlayerGw } from "../season";
import { useData, useSite } from "../site";

interface Line extends PlayerGw { name: string; team: number; position: number }

export default function Gameweeks() {
  const site = useSite();
  const played = site.meta.played;
  const [gw, setGw] = useState(played[played.length - 1]);
  const data = useData<Gameweek>(gw ? gwFile(gw) : null);
  const event = site.meta.events.find((e) => e.id === gw);

  const lines: Line[] = useMemo(() => {
    if (!data) return [];
    return [...totals(data).values()].map((t) => {
      const p = site.player.get(t.element);
      return { ...t, name: p?.web_name ?? String(t.element), team: p?.team ?? 0, position: p?.element_type ?? 0 };
    });
  }, [data, site]);
  const playedLines = lines.filter((l) => l.minutes > 0);
  const midweek = site.midweek.filter((m) => m.gw === gw && m.finished);
  const hasMidweek = playedLines.some((l) => l.midweek !== null);
  const scored = playedLines.filter((l) => l.xp !== null);
  const mae = scored.length ? scored.reduce((s, l) => s + Math.abs(l.points - l.xp!), 0) / scored.length : null;
  const top = [...lines].sort((a, b) => b.points - a.points)[0];
  const captained = event?.most_captained ? site.player.get(event.most_captained) : undefined;
  const captainPts = captained ? lines.find((l) => l.element === captained.id)?.points : undefined;

  const xg = useMemo(() => {
    const byFixture = new Map<number, { home: number; away: number }>();
    if (data) for (const r of rows<GwRow>(data.players)) {
      const f = byFixture.get(r.fixture) ?? { home: 0, away: 0 };
      f[r.was_home ? "home" : "away"] += r.expected_goals ?? 0;
      byFixture.set(r.fixture, f);
    }
    return byFixture;
  }, [data]);

  const scatter = useCallback((width: number) => {
    const maxX = Math.ceil(Math.max(4, ...scored.map((l) => l.xp!)));
    const maxY = Math.ceil(Math.max(4, ...scored.map((l) => l.points)));
    const diag = Math.min(maxX, maxY);
    return Plot.plot({
      ...plotDefaults(width),
      height: 340,
      x: { label: "xP before the deadline", domain: [0, maxX], grid: true },
      y: { label: "Points scored", domain: [Math.min(-2, ...scored.map((l) => l.points)), maxY], grid: true },
      marks: [
        Plot.line([[0, 0], [diag, diag]], { stroke: color.muted, strokeDasharray: "4,4", strokeWidth: 1 }),
        Plot.dot(scored, { x: "xp", y: "points", r: 4, fill: color.s1, fillOpacity: 0.75, stroke: color.surface, strokeWidth: 1 }),
        Plot.tip(scored, Plot.pointer({
          x: "xp", y: "points",
          title: (l: Line) => `${l.name} (${site.team.get(l.team)?.short ?? ""})\n${l.points} points · xP ${pts(l.xp)} · ${l.minutes} min`,
        })),
      ],
    });
  }, [playedLines, scored, site]);

  if (!gw) return <p>No gameweek has been played yet.</p>;

  const columns: Column<Line>[] = [
    { key: "name", label: "Player", value: (l) => l.name, render: (l) => (
      <>{l.name}{l.pre_news ? <> <span className="tag warn" title={l.pre_news}>{l.pre_chance ?? "?"}%</span></> : null}</>) },
    { key: "team", label: "Club", value: (l) => site.team.get(l.team)?.short, render: (l) => <Club id={l.team} /> },
    { key: "pos", label: "Pos", value: (l) => l.position, render: (l) => POSITIONS[l.position] },
    { key: "opp", label: "Opponent", value: (l) => l.opponents.map((o) => site.team.get(o.team)?.short).join(", "),
      render: (l) => l.opponents.map((o) => `${site.team.get(o.team)?.short ?? "?"} (${o.home ? "H" : "A"})`).join(", ") },
    { key: "price", label: "£m", numeric: true, value: (l) => l.price, render: (l) => dec(l.price, 1) },
    { key: "minutes", label: "Mins", numeric: true, value: (l) => l.minutes },
    ...(hasMidweek ? [{ key: "midweek", label: "Midweek", numeric: true, value: (l: Line) => l.midweek,
                        render: (l: Line) => l.midweek === null ? <span className="muted">–</span> : l.midweek,
                        title: "Minutes in his club's cup or European match before this gameweek" } as Column<Line>] : []),
    { key: "points", label: "Points", numeric: true, value: (l) => l.points },
    { key: "xp", label: "xP", numeric: true, value: (l) => l.xp, render: (l) => pts(l.xp),
      title: "The model's expected points before the deadline" },
    { key: "diff", label: "Points − xP", numeric: true, value: (l) => (l.xp === null ? null : l.points - l.xp),
      render: (l) => (l.xp === null ? "–" : <span className={l.points >= l.xp ? "good" : "bad"}>{signed(l.points - l.xp)}</span>) },
    { key: "goals", label: "G", numeric: true, value: (l) => l.goals, title: "Goals" },
    { key: "assists", label: "A", numeric: true, value: (l) => l.assists, title: "Assists" },
    { key: "bonus", label: "Bonus", numeric: true, value: (l) => l.bonus },
    { key: "xg", label: "xG", numeric: true, value: (l) => l.xg, render: (l) => dec(l.xg) },
    { key: "xa", label: "xA", numeric: true, value: (l) => l.xa, render: (l) => dec(l.xa) },
    { key: "dc", label: "DC", numeric: true, value: (l) => l.dc, title: "Defensive contribution: clearances, blocks, interceptions, tackles (and recoveries)" },
    { key: "selected", label: "Owned by", numeric: true, value: (l) => l.selected, render: (l) => compact(l.selected) },
    { key: "transfers", label: "Net transfers", numeric: true, value: (l) => l.transfers, render: (l) => compact(l.transfers),
      title: "Transfers in minus out before this deadline" },
  ];

  return (
    <>
      <h2>Gameweek {gw}</h2>
      <p className="lede">Every result and every player's points, next to what the model expected before the deadline.</p>
      <div className="toolbar">
        <label>
          Gameweek
          <select value={gw} onChange={(e) => setGw(Number(e.target.value))}>
            {[...played].reverse().map((g) => <option key={g} value={g}>GW{g}</option>)}
          </select>
        </label>
        {event && <span className="muted">Deadline {when(event.deadline)}</span>}
      </div>
      {data === undefined && <Loading />}
      {data && (
        <>
          <Tiles tiles={[
            { label: "Average score", value: int(event?.average), note: "across all FPL managers" },
            { label: "Highest score", value: int(event?.highest) },
            { label: "Top player", value: top ? `${top.name} ${top.points}` : "–", note: top && site.team.get(top.team)?.name },
            { label: "Most captained", value: captained ? `${captained.web_name} ${captainPts ?? "–"}` : "–",
              note: captainPts !== undefined ? `${captainPts * 2} with the armband` : undefined },
            { label: "Model error", value: mae === null ? "–" : pts(mae),
              note: `average miss in points, ${scored.length} players who played` },
          ]} />

          <h3>Results</h3>
          <div className="fixtures">
            {data.fixtures.map((f) => {
              const x = xg.get(f.id);
              return (
                <div className="fixture" key={f.id}>
                  <span><Club id={f.home} /></span>
                  <span className="score">{f.home_score ?? "–"} – {f.away_score ?? "–"}</span>
                  <span className="away"><Club id={f.away} /></span>
                  {x && <span className="xg">xG {dec(x.home)} – {dec(x.away)}</span>}
                </div>
              );
            })}
          </div>

          {midweek.length > 0 && (
            <>
              <h3>Midweek before GW{gw}</h3>
              <div className="fixtures">
                {midweek.map((m) => (
                  <div className="fixture" key={m.match_id}>
                    <span>{m.home_code !== null && site.teamByCode.has(m.home_code) ? <Club code={m.home_code} /> : side(site, m, true)}</span>
                    <span className="score">{m.home_score ?? "–"} – {m.away_score ?? "–"}</span>
                    <span className="away">{m.away_code !== null && site.teamByCode.has(m.away_code) ? <Club code={m.away_code} /> : side(site, m, false)}</span>
                    <span className="xg">{competition(m.tournament).name} · {when(m.kickoff)}</span>
                  </div>
                ))}
              </div>
            </>
          )}

          <div className="card">
            <h3 style={{ marginTop: 0 }}>Points against xP</h3>
            <p className="note" style={{ marginTop: 0 }}>
              One dot per player who played. Dots above the dashed line beat their xP. Hover for names.
            </p>
            <Chart make={scatter} height={340} ariaLabel={`Points against xP for GW${gw}`} />
          </div>

          <h3>Players who played</h3>
          <Table columns={columns} data={playedLines} sort="points" rowKey={(l) => l.element} limit={40} />
          <Note>
            xP here is {data.xp_source}. A red % tag is the injury flag FPL showed before the deadline, where one
            was recorded. Prices (£m) are as they stood during the gameweek.
            {hasMidweek && <> Midweek is his minutes in his club's cup or European match before this gameweek
              (0: in the squad but not used; – : his club had no midweek match).</>}
          </Note>
        </>
      )}
    </>
  );
}
