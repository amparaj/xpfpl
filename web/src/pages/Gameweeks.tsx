import * as Plot from "@observablehq/plot";
import { useCallback, useMemo, useState } from "react";
import { color } from "../colors";
import { Chart, Club, Note, Loading, Opponent, Table, Tiles, plotDefaults, type Column } from "../components/ui";
import { competition, side } from "../midweek";
import { gwFile, rows, type Gameweek, type GwRow, type ModelTeam } from "../data";
import { POSITIONS, compact, dec, int, pts, signed, when } from "../format";
import { totals, type PlayerGw } from "../season";
import { SiteContext, gameweekHref, useData, useHash, useSeason, useSite } from "../site";
import Match from "./Match";

interface Line extends PlayerGw { name: string; team: number; position: number }

/** #gameweeks shows the latest played gameweek, #gameweeks/6 GW6, and #gameweeks/6/53 fixture 53's match page.
 * An earlier season goes first: #gameweeks/2025-26/12/115 (its own copy of the data, seasons/2025-26/). */
export default function Gameweeks() {
  const current = useSite();
  const parts = useHash().split("/").slice(1);
  const season = /^\d{4}-\d{2}$/.test(parts[0] ?? "") ? parts.shift()! : null;
  const site = useSeason(season);
  if (site === undefined) return <Loading />;
  if (site === null) return <p>That season isn't on the site. <a href="#gameweeks">Back to {current.meta.season}</a></p>;
  const played = site.meta.played;
  const [wanted, fixture] = parts.map(Number);
  const gw = played.includes(wanted) ? wanted : played[played.length - 1];
  return (
    <SiteContext.Provider value={site}>
      {gw && fixture ? <Match key={`${site.meta.season}/${fixture}`} gw={gw} fixture={fixture} />
        : <Week gw={gw} seasons={current.meta.seasons ?? [current.meta.season]} />}
    </SiteContext.Provider>
  );
}

function Week({ gw, seasons }: { gw: number; seasons: string[] }) {
  const site = useSite();
  const played = site.meta.played;
  const past = !!site.meta.past;
  const setGw = (g: number) => { window.location.hash = gameweekHref(site, g); };
  const setSeason = (s: string) => { window.location.hash = s === seasons[seasons.length - 1] ? "gameweeks" : `gameweeks/${s}`; };
  const data = useData<Gameweek>(gw ? site.root + gwFile(gw) : null);
  const team = useData<ModelTeam>(past ? null : "modelteam.json");      // the Model's Team is this season's only
  const teamWeek = team?.gameweeks.find((w) => w.gw === gw);
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
  const hasXp = lines.some((l) => l.xp !== null);
  const mae = scored.length ? scored.reduce((s, l) => s + Math.abs(l.points - l.xp!), 0) / scored.length : null;
  // Everyone with a forecast, including those who didn't play: their xP already allowed for that
  // chance, so leaving them out would make the forecasts look too low.
  const forecast = lines.filter((l) => l.xp !== null);
  const xpSum = forecast.reduce((s, l) => s + l.xp!, 0);
  const pointsSum = forecast.reduce((s, l) => s + l.points, 0);
  // The model's own top forecasts this week (whoever played), and what they scored.
  const topForecasts = [...lines].filter((l) => l.xp !== null).sort((a, b) => b.xp! - a.xp!).slice(0, 10);
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

  const [highlight, setHighlight] = useState(0);        // a club id to pick out on the scatter (0: none)
  const clubs = useMemo(() => [...new Set(scored.map((l) => l.team))]
    .map((id) => site.team.get(id)).filter((t) => t !== undefined)
    .sort((a, b) => a.name.localeCompare(b.name)), [scored, site]);
  const scatter = useCallback((width: number) => {
    const maxX = Math.ceil(Math.max(4, ...scored.map((l) => l.xp!)));
    const maxY = Math.ceil(Math.max(4, ...scored.map((l) => l.points)));
    const diag = Math.min(maxX, maxY);
    const club = (l: Line) => site.meta.club_colours[site.team.get(l.team)?.short ?? ""]?.[0] ?? color.muted;
    const picked = (l: Line) => !highlight || l.team === highlight;
    // The picked club's dots drawn last, so they sit on top.
    const ordered = [...scored].sort((a, b) => Number(picked(a)) - Number(picked(b)));
    return Plot.plot({
      ...plotDefaults(width),
      height: 340,
      x: { label: "xP before the deadline", domain: [0, maxX], grid: true },
      y: { label: "Points scored", domain: [Math.min(-2, ...scored.map((l) => l.points)), maxY], grid: true },
      marks: [
        Plot.line([[0, 0], [diag, diag]], { stroke: color.muted, strokeDasharray: "4,4", strokeWidth: 1 }),
        // An outline in the text colour keeps the dark clubs (Fulham, Newcastle) visible in dark mode.
        Plot.dot(ordered, {
          x: "xp", y: "points", r: (l: Line) => (highlight && picked(l) ? 5.5 : 4.5), fill: club,
          fillOpacity: (l: Line) => (picked(l) ? 0.9 : 0.12), stroke: color.ink,
          strokeOpacity: (l: Line) => (picked(l) ? 0.45 : 0.1), strokeWidth: 0.75,
        }),
        Plot.tip(scored, Plot.pointer({
          x: "xp", y: "points",
          title: (l: Line) => `${l.name} (${site.team.get(l.team)?.short ?? ""})\n${l.points} points · xP ${pts(l.xp)} · ${l.minutes} min`,
        })),
      ],
    });
  }, [scored, site, highlight]);

  if (!gw) return <p>No gameweek has been played yet.</p>;

  const columns: Column<Line>[] = [
    { key: "name", label: "Player", value: (l) => l.name, render: (l) => (
      <>{l.name}{l.pre_news ? <> <span className="tag warn" title={l.pre_news}>{l.pre_chance ?? "?"}%</span></> : null}</>) },
    { key: "team", label: "Club", value: (l) => site.team.get(l.team)?.short, render: (l) => <Club id={l.team} /> },
    { key: "pos", label: "Pos", value: (l) => l.position, render: (l) => POSITIONS[l.position] },
    { key: "opp", label: "Opponent", value: (l) => l.opponents.map((o) => site.team.get(o.team)?.short).join(", "),
      render: (l) => l.opponents.map((o, i) => <span key={i}>{i > 0 && ", "}<Opponent id={o.team} home={o.home} /></span>) },
    { key: "price", label: "£m", numeric: true, value: (l) => l.price, render: (l) => dec(l.price, 1) },
    { key: "minutes", label: "Mins", numeric: true, value: (l) => l.minutes },
    ...(hasMidweek ? [{ key: "midweek", label: "Midweek", numeric: true, value: (l: Line) => l.midweek,
                        render: (l: Line) => l.midweek === null ? <span className="muted">–</span> : l.midweek,
                        title: "Minutes in his club's cup or European match before this gameweek" } as Column<Line>] : []),
    { key: "points", label: "Points", numeric: true, value: (l) => l.points },
    { key: "xp", label: "xP", numeric: true, value: (l) => l.xp, render: (l) => pts(l.xp),
      title: "This model's expected points before the deadline" },
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

  const forecastColumns: Column<Line>[] = [
    { key: "name", label: "Player", value: (l) => l.name },
    { key: "team", label: "Club", value: (l) => site.team.get(l.team)?.short, render: (l) => <Club id={l.team} /> },
    { key: "pos", label: "Pos", value: (l) => l.position, render: (l) => POSITIONS[l.position] },
    { key: "xp", label: "xP", numeric: true, value: (l) => l.xp, render: (l) => pts(l.xp), title: "This model's expected points before the deadline" },
    { key: "points", label: "Points", numeric: true, value: (l) => l.points },
    { key: "diff", label: "Points − xP", numeric: true, value: (l) => (l.xp === null ? null : l.points - l.xp),
      render: (l) => (l.xp === null ? "–" : <span className={l.points >= l.xp ? "good" : "bad"}>{signed(l.points - l.xp)}</span>) },
    { key: "minutes", label: "Mins", numeric: true, value: (l) => l.minutes },
  ];

  return (
    <>
      <h2>Gameweek {gw}{past && `, ${site.meta.season}`}</h2>
      <p className="lede">
        {hasXp || !data ? `Every result and every player's points, next to what this ${site.meta.model} model expected before the deadline.`
          : "Every result and every player's points. No forecast was saved before this deadline, so there's no xP here."}
      </p>
      <div className="toolbar">
        {seasons.length > 1 && (
          <label>
            Season
            <select value={site.meta.season} onChange={(e) => setSeason(e.target.value)}>
              {[...seasons].reverse().map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
        )}
        <label>
          Gameweek
          <select value={gw} onChange={(e) => setGw(Number(e.target.value))}>
            {[...played].reverse().map((g) => <option key={g} value={g}>GW{g}</option>)}
          </select>
        </label>
        {/* Without FPL's gameweek summary (a season archived from vaastav) the first kick-off stands in. */}
        {event && <span className="muted">{past && event.average == null ? "First kick-off" : "Deadline"} {when(event.deadline)}</span>}
      </div>
      {data === undefined && <Loading />}
      {data && (
        <>
          <Tiles tiles={[
            // An earlier season archived from vaastav has no FPL gameweek summary (average, highest, captain).
            ...(event?.average != null || !past ? [{ label: "Average score", value: int(event?.average), note: "across all FPL managers" }] : []),
            ...(event?.highest != null || !past ? [{ label: "Highest score", value: int(event?.highest) }] : []),
            { label: "Top player", value: top ? `${top.name} ${top.points}` : "–", note: top && <Club id={top.team} /> },
            ...(past ? [
              { label: "Goals", value: int(data.fixtures.reduce((s, f) => s + (f.home_score ?? 0) + (f.away_score ?? 0), 0)),
                note: `in ${data.fixtures.length} matches` },
              { label: "Players who played", value: int(playedLines.length) },
            ] : []),
            ...(captained || !past ? [{ label: "Most captained", value: captained ? `${captained.web_name} ${captainPts ?? "–"}` : "–",
              note: captainPts !== undefined ? `${captainPts * 2} with the armband` : undefined }] : []),
            ...(hasXp ? [{ label: "Model error", value: mae === null ? "–" : pts(mae),
              note: `average miss in points, ${scored.length} players who played` }] : []),
            ...(forecast.length ? [{ label: "Forecast vs scored", value: `${int(xpSum)} → ${int(pointsSum)}`,
              note: `all ${forecast.length} players with a forecast, played or not: ${signed(pointsSum - xpSum, 0)} points` }] : []),
            ...(teamWeek?.forecast != null ? [{ label: "This Model's Team", value: `Scored ${teamWeek.gross ?? "–"}`,
              note: <>against a forecast of {int(teamWeek.forecast)} (before transfer hits); <a href="#model-team">see this model's team</a></> }] : []),
          ]} />

          <h3>Results</h3>
          <p className="note" style={{ marginTop: 0 }}>Pick a result for its shot map, how the match went, the team stats and every player's numbers.</p>
          <div className="fixtures">
            {data.fixtures.map((f) => {
              const x = xg.get(f.id);
              return (
                <a className="fixture" key={f.id} href={gameweekHref(site, gw, f.id)}
                   aria-label={`${site.team.get(f.home)?.name} ${f.home_score ?? ""} ${site.team.get(f.away)?.name} ${f.away_score ?? ""}: match details`}>
                  <span><Club id={f.home} /></span>
                  <span className="score">{f.home_score ?? "–"} – {f.away_score ?? "–"}</span>
                  <span className="away"><Club id={f.away} /></span>
                  <span className="xg">{x ? `xG ${dec(x.home)} – ${dec(x.away)}` : "Match details"}</span>
                </a>
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

          {hasXp && <div className="card">
            <h3 style={{ marginTop: 0 }}>Points against xP</h3>
            <p className="note" style={{ marginTop: 0 }}>
              One dot per player who played, in their club's colour. Dots above the dashed line beat their xP.
            </p>
            <div className="toolbar">
              <label>Highlight a club{" "}
                <select value={highlight} onChange={(e) => setHighlight(Number(e.target.value))}>
                  <option value={0}>None</option>
                  {clubs.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
                </select>
              </label>
            </div>
            <Chart make={scatter} height={340} ariaLabel={`Points against xP for GW${gw}`} />
          </div>}

          {topForecasts.length > 0 && (
            <>
              <h3>This model's top forecasts</h3>
              <p className="note" style={{ marginTop: 0 }}>The ten highest xP before the deadline, and what they scored.</p>
              <Table columns={forecastColumns} data={topForecasts} sort="xp" rowKey={(l) => l.element} />
            </>
          )}

          <h3>Players who played</h3>
          <Table columns={hasXp ? columns : columns.filter((c) => c.key !== "xp" && c.key !== "diff")}
                 data={playedLines} sort="points" rowKey={(l) => l.element} limit={40} />
          <Note>
            {hasXp && <>xP here is {data.xp_source}. </>}A red % tag is the injury flag FPL showed before the deadline, where one
            was recorded. Prices (£m) are as they stood during the gameweek.
            {hasMidweek && <> Midweek is his minutes in his club's cup or European match before this gameweek
              (0: in the squad but not used; – : his club had no midweek match).</>}
          </Note>
        </>
      )}
    </>
  );
}
