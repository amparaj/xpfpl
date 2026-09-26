import * as Plot from "@observablehq/plot";
import { useCallback, useMemo, useState } from "react";
import { color } from "../colors";
import { band } from "../components/Simulation";
import { Chart, Club, Legend, Loading, MidweekBadge, Note, Segmented, Table, Tiles, plotDefaults, type Column, type TileProps } from "../components/ui";
import { rows, type ModelTeam, type NextGw, type Player } from "../data";
import { POSITIONS, dec, money, pct, pts, signed, when } from "../format";
import { ROTATION, clubMidweek, competition } from "../midweek";
import { oddsUrl, type OddsSnapshot } from "../polymarket";
import { matchPlayer, ours } from "../ratings";
import { useData, useSite, type Site } from "../site";

interface Forecast {
  element: number; xp_total: number; p_play?: number | null; xmins?: number | null;
  rotation?: string | null; rotation_factor?: number | null;
  /** The next gameweek's Monte Carlo: 10th/50th/90th percentile of his simulated points, chance of 10+ and of 2 or fewer. */
  pts_p10?: number | null; pts_p50?: number | null; pts_p90?: number | null; p_haul?: number | null; p_blank?: number | null;
  [xp: `xp_${number}`]: number;
}
type Row = Forecast & { player: Player };

/** A club's fixtures in one gameweek: the opponent, home or away, and our chance of winning. */
interface Match { opponent: number; home: boolean; win: number | null; gf: number | null; ga: number | null }

function clubMatches(site: Site, team: number, gw: number): Match[] {
  return site.fixtures.filter((f) => f.gw === gw && (f.home === team || f.away === team)).map((f) => {
    const home = f.home === team;
    const code = (id: number) => site.team.get(id)!.code;
    const o = ours(site, code(f.home), code(f.away));
    return { opponent: home ? f.away : f.home, home, win: home ? o.ours_home_win : o.ours_away_win,
             gf: home ? o.ours_home : o.ours_away, ga: home ? o.ours_away : o.ours_home };
  });
}

function Opponents({ matches }: { matches: Match[] }) {
  const site = useSite();
  if (!matches.length) return <span className="muted">blank</span>;
  return <>{matches.map((m, i) => (
    <span key={i}>{i > 0 && ", "}{site.team.get(m.opponent)?.short} ({m.home ? "H" : "A"})</span>
  ))}</>;
}

export default function NextGameweek() {
  const site = useSite();
  const next = useData<NextGw>("next.json");
  const team = useData<ModelTeam>("modelteam.json");
  const odds = useData<OddsSnapshot>(oddsUrl());
  const [position, setPosition] = useState(0);

  const players: Row[] = useMemo(() => rows<Forecast>(next?.players)
    .map((f) => ({ ...f, player: site.player.get(f.element)! })).filter((r) => r.player), [next, site]);

  const gw = next?.gw ?? 0;
  const first = `xp_${gw}` as const;
  const captains = useMemo(() => [...players].sort((a, b) => b[first] - a[first]).slice(0, 10), [players, first]);

  const ranged = captains.some((r) => r.pts_p90 != null);
  const captainChart = useCallback((width: number) => {
    const label = (r: Row) => `${r.player.web_name} (${site.team.get(r.player.team)?.short})`;
    if (ranged) {
      const top = Math.max(...captains.map((r) => r.pts_p90 ?? 0), 1);
      return Plot.plot({
        ...plotDefaults(width),
        marginLeft: 150,
        marginRight: 56,
        height: captains.length * 28 + 40,
        x: { label: `GW${gw} points`, grid: true, domain: [0, top + 1] },
        y: { label: null, domain: captains.map(label) },
        marks: [
          Plot.ruleY(captains, { y: label, x1: "pts_p10", x2: "pts_p90", stroke: color.s1, strokeWidth: 6, strokeOpacity: 0.35, strokeLinecap: "round" }),
          Plot.dot(captains, { x: first, y: label, r: 5, fill: color.s1, stroke: color.surface, strokeWidth: 2 }),
          Plot.text(captains, { x: top + 1, y: label, text: (r: Row) => `${Math.round((r.p_haul ?? 0) * 100)}% 10+`, dx: 6, textAnchor: "start", fill: color.ink2 }),
          Plot.tip(captains, Plot.pointerY({ x: first, y: label, title: (r: Row) =>
            `${label(r)}: ${pts(r[first])} xP\nmiddle 80% of simulated weeks: ${band(r.pts_p10, r.pts_p90)} points\n` +
            `10+: ${pct(r.p_haul)} · 2 or fewer: ${pct(r.p_blank)}` })),
        ],
      });
    }
    return Plot.plot({
      ...plotDefaults(width),
      marginLeft: 150,
      height: captains.length * 28 + 40,
      x: { label: `xP in GW${gw}`, grid: true, nice: true },
      y: { label: null, domain: captains.map(label) },
      marks: [
        Plot.barX(captains, { x: first, y: label, fill: color.s1, rx2: 4, insetTop: 4, insetBottom: 4 }),
        Plot.text(captains, { x: first, y: label, text: (r: Row) => pts(r[first]), dx: 6, textAnchor: "start", fill: color.ink2 }),
        Plot.tip(captains, Plot.pointerY({ x: first, y: label, title: (r: Row) => `${label(r)}: ${pts(r[first])} xP` })),
      ],
    });
  }, [captains, first, gw, site, ranged]);

  // Every club's fixtures over the forecast's horizon, with our chance of winning each.
  const clubs = useMemo(() => site.meta.teams.map((t) => {
    const byGw = (next?.gameweeks ?? []).map((g) => clubMatches(site, t.id, g));
    const wins = byGw.flat().map((m) => m.win).filter((w): w is number => w !== null);
    return { team: t.id, byGw, avg: wins.length ? wins.reduce((a, b) => a + b, 0) / wins.length : null };
  }), [next, site]);

  const out = useMemo(() => {
    if (!odds) return null;
    const nextMatches = new Set(odds.matches.filter((m) => new Date(m.kickoff).getTime() > Date.now()).map((m) => m.slug));
    return odds.scorers.filter((s) => nextMatches.has(s.slug) && s.p <= site.meta.out_threshold)
      .map((s) => ({ ...s, fpl: matchPlayer(site, s) }));
  }, [odds, site]);

  if (next === undefined) return <Loading />;
  if (next === null) {
    return (
      <>
        <h2>Next Gameweek</h2>
        <p>No forecast has been saved for GW{site.meta.next_gw} yet. It appears once the model has been run for the gameweek.</p>
      </>
    );
  }

  const horizon = next.gameweeks;
  const shown = players.filter((r) => !position || r.player.element_type === position)
    .sort((a, b) => b.xp_total - a.xp_total).slice(0, 30);
  const hasPlay = players.some((r) => r.p_play != null);
  const hasRotation = players.some((r) => r.rotation != null);
  const hasRanges = players.some((r) => r.pts_p90 != null);
  const midweekAhead = horizon.some((g) => site.midweek.some((m) => m.gw === g));

  const playerColumns: Column<Row>[] = [
    { key: "name", label: "Player", value: (r) => r.player.web_name },
    { key: "club", label: "Club", value: (r) => site.team.get(r.player.team)?.short, render: (r) => <Club id={r.player.team} /> },
    { key: "pos", label: "Pos", value: (r) => r.player.element_type, render: (r) => POSITIONS[r.player.element_type] },
    { key: "price", label: "Price", numeric: true, value: (r) => r.player.now_cost, render: (r) => money(r.player.now_cost) },
    { key: "opp", label: `GW${gw}`, value: (r) => clubMatches(site, r.player.team, gw).map((m) => m.opponent).join(),
      render: (r) => <><MidweekBadge matches={clubMidweek(site, r.player.team, gw)} /><Opponents matches={clubMatches(site, r.player.team, gw)} /></>,
      title: `Opponent in GW${gw}` },
    ...(hasRotation ? [{ key: "midweek", label: "Midweek", numeric: true, value: (r: Row) => r.rotation_factor,
                         render: (r: Row) => r.rotation ? (
                           <span className={(r.rotation_factor ?? 1) >= 1 ? "good" : "bad"} title={ROTATION[r.rotation] ?? r.rotation}>
                             ×{dec(r.rotation_factor)}
                           </span>) : <span className="muted">–</span>,
                         title: "What his minutes in the midweek cup or European match did to his GW xP" } as Column<Row>] : []),
    ...(hasPlay ? [{ key: "play", label: "Plays", numeric: true, value: (r: Row) => r.p_play, render: (r: Row) => pct(r.p_play),
                     title: "The model's chance he plays in the next gameweek" } as Column<Row>] : []),
    ...horizon.map((g): Column<Row> => ({ key: `xp${g}`, label: `GW${g}`, group: "xP", numeric: true,
                                          value: (r) => r[`xp_${g}`], render: (r) => pts(r[`xp_${g}`]) })),
    { key: "total", label: "Total", group: "xP", numeric: true, value: (r) => r.xp_total, render: (r) => <strong>{pts(r.xp_total)}</strong>,
      title: `Expected points over GW${horizon[0]}–${horizon[horizon.length - 1]}` },
    ...(hasRanges ? [
      { key: "range", label: "Range", group: `GW${gw} simulated`, numeric: true, value: (r: Row) => r.pts_p90,
        render: (r: Row) => band(r.pts_p10, r.pts_p90), title: `The middle 80% of his simulated GW${gw} points` } as Column<Row>,
      { key: "haul", label: "10+", group: `GW${gw} simulated`, numeric: true, value: (r: Row) => r.p_haul, render: (r: Row) => pct(r.p_haul),
        title: `His chance of 10+ points in GW${gw}` } as Column<Row>,
      { key: "blank", label: "≤2", group: `GW${gw} simulated`, numeric: true, value: (r: Row) => r.p_blank, render: (r: Row) => pct(r.p_blank),
        title: `His chance of 2 points or fewer in GW${gw}, not playing included` } as Column<Row>,
    ] : []),
  ];

  type Club = (typeof clubs)[number];
  const fixtureColumns: Column<Club>[] = [
    { key: "club", label: "Club", value: (c) => site.team.get(c.team)?.short, render: (c) => <Club id={c.team} /> },
    ...horizon.map((g, i): Column<Club> => ({
      key: `gw${g}`, label: `GW${g}`, value: (c) => c.byGw[i].map((m) => m.win ?? 0).reduce((a, b) => a + b, 0),
      render: (c) => c.byGw[i].length ? (
        <span className="fixture-cells">
          <MidweekBadge matches={clubMidweek(site, c.team, g)} />
          {c.byGw[i].map((m, j) => (
            <span key={j} className="fixture-cell" style={{ background: `color-mix(in srgb, var(--s1) ${Math.round((m.win ?? 0) * 80)}%, transparent)` }}
                  title={m.gf === null ? undefined : `Expected goals ${m.gf.toFixed(1)}–${m.ga!.toFixed(1)}, win chance ${pct(m.win)}`}>
              {site.team.get(m.opponent)?.short} ({m.home ? "H" : "A"}) {pct(m.win)}
            </span>
          ))}
        </span>
      ) : <span className="muted">blank</span>,
    })),
    { key: "avg", label: "Average", numeric: true, value: (c) => c.avg, render: (c) => pct(c.avg), title: "Average chance of winning over these fixtures" },
  ];

  const upcoming = team?.next?.gw === gw ? team.next : null;
  const lastWeek = team?.gameweeks.length ? team.gameweeks[team.gameweeks.length - 1] : null;
  const tiles: TileProps[] = [
    ...(upcoming ? [{ label: "The Model's Team", value: `${pts(upcoming.forecast ?? upcoming.xp)} xP`,
      note: <>its forecast for GW{gw}, captain doubled{upcoming.simulation
        ? <>; {band(upcoming.simulation.points.p10, upcoming.simulation.points.p90)} in 4 simulated weeks out of 5</> : null};{" "}
        <a href="#model-team">see the team</a></> }] : []),
    ...(captains.length ? [{ label: "Top forecast", value: `${captains[0].player.web_name} ${pts(captains[0][first])}`,
      note: `xP for GW${gw} (${site.team.get(captains[0].player.team)?.short ?? ""})` }] : []),
    ...(lastWeek?.forecast != null ? [{ label: `Last time (GW${lastWeek.gw})`, value: `${pts(lastWeek.forecast)} → ${lastWeek.gross}`,
      note: `the Model's Team: forecast → scored, ${signed((lastWeek.gross ?? 0) - lastWeek.forecast, 1)}` }] : []),
  ];

  return (
    <>
      <h2>Next Gameweek: GW{gw}</h2>
      <p className="lede">
        The model's forecast for GW{gw}, whose deadline is {when(next.deadline)}: who is expected to score the most, the captain
        picks, and how each club's next {horizon.length === 1 ? "fixture looks" : `${horizon.length} gameweeks look`}.
      </p>
      {tiles.length > 0 && <Tiles tiles={tiles} />}

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Captain picks</h3>
        {ranged && <Legend items={[{ label: "xP", color: color.s1, kind: "dot" }, { label: "Middle 80% of simulated scores", color: `color-mix(in srgb, ${color.s1} 35%, transparent)` }]} />}
        <Chart make={captainChart} height={captains.length * 28 + 40} ariaLabel={`The ten players with the highest xP in GW${gw}`} />
        <p className="note">
          The ten highest xP for GW{gw} alone. A captain scores double.
          {ranged && <> The bar is each player's middle 80% of scores when the gameweek was simulated thousands of times, and the
            number is his chance of 10+: two picks with the same xP can carry very different risk (<a href="#about">how it works</a>).</>}
        </p>
      </div>

      <h3>Top players</h3>
      <div className="toolbar">
        <Segmented label="Position" value={position} onChange={setPosition}
                   options={[{ value: 0, label: "All" }, ...[1, 2, 3, 4].map((p) => ({ value: p, label: POSITIONS[p] }))]} />
      </div>
      <Table columns={playerColumns} data={shown} sort="total" rowKey={(r) => r.element} />
      <Note>
        The 30 highest expected points (xP) over GW{horizon[0]}–{horizon[horizon.length - 1]}, already cut for injury flags.
        {hasRanges && <> Simulated: the middle 80% of his GW{gw} points, and his chances of 10+ and of 2 or fewer, from playing the
          gameweek thousands of times.</>}
        {hasRotation && <> Midweek: a player whose club played a cup or European match this week has his GW{gw} xP
          multiplied by what his own minutes in it have meant in the past, most of all for squad players (hover for his group;
          see <a href="#about">About</a>).</>}
      </Note>

      <h3>Fixtures</h3>
      <Table columns={fixtureColumns} data={clubs} sort="avg" rowKey={(c) => c.team} />
      <Note>
        Each club's chance of winning, from the model's own club ratings (the darker, the likelier).
        {midweekAhead && <> A badge such as <span className="tag cup">{competition("champions-league").short} Tue</span>
          marks a cup or European match in the week before that gameweek (hover for the opponent).</>}
      </Note>

      <h3>Ruled out by the betting markets</h3>
      {out === undefined ? <Loading />
        : !out || !odds?.scorers.length ? <p className="muted">Goalscorer markets for GW{gw} aren't open yet. They usually open a day or two before kick-off.</p>
        : !out.length ? <p className="muted">No player's goalscorer odds are down at {pct(site.meta.out_threshold)} or less.</p>
        : (
          <ul>
            {out.map((s) => (
              <li key={`${s.slug}-${s.player}`}>
                {s.fpl ? <><Club id={s.fpl.team} /> {s.fpl.web_name}</> : s.player}: {pct(s.p, 1)} to score
                {s.fpl?.forecast != null && <span className="muted"> (xP {pts(s.fpl.forecast)})</span>}
              </li>
            ))}
          </ul>
        )}
      <Note>Goalscorer odds of {pct(site.meta.out_threshold)} or less almost always mean the player has been ruled out, which the FPL flags may not show yet.</Note>
    </>
  );
}
