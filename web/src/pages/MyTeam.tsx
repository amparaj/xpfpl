// My Team: the team I'm playing in the gameweek in progress (until it's finished) or planning for the
// next one, and how it could do this week and the next few, from the Monte Carlo. It never shows
// results: once FPL marks the gameweek finished the export moves on, and Past Gameweeks has how it went.
//
// The numbers are for anyone. Before the deadline the team itself (players, formation, captains, chip,
// transfers) is encrypted, and the secret word opens it in the browser (seal.ts); after the deadline
// FPL shows it anyway, so it comes in plain.

import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Pitch } from "../components/Pitch";
import { CaptainOddsTable, ScoreChart, band } from "../components/Simulation";
import { Club, Loading, Note, Segmented, Table, Tiles, type Column, type TileProps } from "../components/ui";
import { rows, type MyTeam, type MyTeamPrivate, type MyWeek, type MyWeekNumbers, type PlanSettings, type Player } from "../data";
import { POSITIONS, int, money, pct, pts, signed, when } from "../format";
import { rememberWord, rememberedWord, unseal } from "../seal";
import { useData, useSite } from "../site";

const CHIP_NAMES: Record<string, string> = { wildcard: "Wildcard", freehit: "Free Hit", bboost: "Bench Boost", "3xc": "Triple Captain" };

interface Squad {
  element: number; xp_total: number; pts_p10?: number | null; pts_p90?: number | null;
  p_haul?: number | null; p_blank?: number | null; p_play?: number | null;
  [xp: `xp_${number}`]: number;
}
type SquadRow = Squad & { player: Player; role: string; order: number };

/** The team: the data's own after the deadline, else what the secret word unlocked (remembered in this browser). */
function useTeam(data: MyTeam | null | undefined) {
  const [unlocked, setUnlocked] = useState<MyTeamPrivate | null>(null);
  const [state, setState] = useState<"idle" | "busy" | "wrong">("idle");
  const sealed = data?.sealed ?? null;

  const unlock = async (word: string, remember: boolean) => {
    if (!sealed || !word.trim()) return;
    setState("busy");
    const opened = await unseal<MyTeamPrivate>(sealed, word);
    setUnlocked(opened);
    setState(opened ? "idle" : "wrong");
    if (opened && remember) rememberWord(word);
    if (!opened && !remember) rememberWord(null);          // a remembered word that no longer works
  };
  useEffect(() => {
    const word = rememberedWord();
    if (sealed && word) void unlock(word, false);
  }, [sealed]); // eslint-disable-line react-hooks/exhaustive-deps

  const lock = () => { rememberWord(null); setUnlocked(null); setState("idle"); };
  return { team: data?.private ?? unlocked, unlocked: !!unlocked, state, unlock, lock };
}

/** The week's transfers, one "out → in" row each (both sorted by position when saved). */
function Transfers({ moves, hits }: { moves: { out: number; in: number }[]; hits: number }) {
  const site = useSite();
  const side = (id: number) => {
    const p = site.player.get(id);
    return <>{p ? <Club id={p.team} /> : null}<span className="transfer-name">{p?.web_name ?? id}</span></>;
  };
  return (
    <>
      <h4>{moves.length} transfer{moves.length === 1 ? "" : "s"}{hits ? `, −${4 * hits} in penalties` : ""}</h4>
      <ol className="transfers">
        {moves.map((t) => (
          <li key={`${t.out}-${t.in}`} className="transfer">
            <span className="transfer-pos">{POSITIONS[site.player.get(t.in)?.element_type ?? 0] ?? ""}</span>
            <span className="transfer-out">{side(t.out)}</span>
            <span className="transfer-arrow" aria-label="replaced by">→</span>
            <span className="transfer-in">{side(t.in)}</span>
          </li>
        ))}
      </ol>
    </>
  );
}

/** The Plan Ahead settings the team was saved with, to set the same scenario up again. */
function Settings({ settings, locked }: { settings: PlanSettings; locked: boolean }) {
  const site = useSite();
  const names = (ids: number[]) => (ids.length ? ids.map((id) => site.player.get(id)?.web_name ?? id).join(", ") : "–");
  const s = settings;
  const items: [string, string][] = [
    ["Horizon", `${s.horizon} gameweek${s.horizon === 1 ? "" : "s"}`],
    ["Model", s.model],
    ["Chip", s.chip ? CHIP_NAMES[s.chip] ?? s.chip : "None"],
    ["Free transfers", `${s.free_transfers}${s.free_transfers_estimated ? " (estimated)" : ""}`],
    ["Bank", money(s.bank)],
    ["Max penalties", String(s.max_hits)],
    ["Week-by-week planning", s.plan_transfers ? "On" : "Off"],
    ["Value price rises", s.value_prices ? "On" : "Off"],
    ["Always pick", names(s.must_have)],
    ["Never pick", names(s.banned)],
  ];
  return (
    <>
      <h4>{locked ? "Last saved in Plan Ahead with" : "Planned in Plan Ahead with"}</h4>
      <dl className="settings">
        {items.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}
      </dl>
    </>
  );
}

function Unlock({ gw, busy, wrong, onUnlock }: { gw: number; busy: boolean; wrong: boolean; onUnlock: (word: string) => void }) {
  const [word, setWord] = useState("");
  const submit = (e: FormEvent) => { e.preventDefault(); onUnlock(word); };
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>The team is hidden until the GW{gw} deadline</h3>
      <p style={{ marginTop: 0 }}>
        The players, formation, captain, vice-captain, chip and transfers stay private until the deadline. The numbers above are the
        team's forecast.
      </p>
      <form className="toolbar" onSubmit={submit}>
        <label>
          Secret word
          <input type="password" value={word} onChange={(e) => setWord(e.target.value)} autoComplete="current-password" />
        </label>
        <button type="submit" disabled={busy || !word.trim()}>{busy ? "Checking…" : "Show the team"}</button>
        {wrong && <span className="bad">That's not it.</span>}
      </form>
    </div>
  );
}

export default function MyTeamPage() {
  const site = useSite();
  const data = useData<MyTeam>("myteam.json");
  const { team, unlocked, state, unlock, lock } = useTeam(data);
  const [chosen, setChosen] = useState<number | null>(null);
  const name = (id: number) => site.player.get(id)?.web_name ?? String(id);

  const squad: SquadRow[] = useMemo(() => {
    if (!team) return [];
    const first = team.weeks[0];
    const order = [...first.lineup, ...first.bench];
    return rows<Squad>(team.players).map((r) => {
      const i = order.indexOf(r.element);
      const role = r.element === first.captain ? "Captain" : r.element === first.vice ? "Vice" : i < 0 ? "–" : i < 11 ? "XI" : `Bench ${i - 10}`;
      return { ...r, player: site.player.get(r.element)!, role, order: i < 0 ? 99 : i };
    }).filter((r) => r.player);
  }, [team, site]);

  if (data === undefined) return <Loading />;
  if (data === null) {
    return (
      <>
        <h2>My Team</h2>
        <p>No team to show for GW{site.meta.next_gw} yet. It appears once this model has made its forecast for the gameweek.</p>
      </>
    );
  }

  const first = data.weeks[0];
  const shown = chosen ?? data.gw;
  const numbers: MyWeekNumbers = data.weeks.find((w) => w.gw === shown) ?? first;
  const week: MyWeek | undefined = team?.weeks.find((w) => w.gw === shown);
  const later = data.gameweeks.length > 1;
  const span = later ? `GW${data.gameweeks[0]}–${data.gameweeks[data.gameweeks.length - 1]}` : `GW${data.gw}`;
  const byId = new Map(squad.map((r) => [r.element, r]));
  const a = data.against;
  const chip = team?.chip ?? null;
  const described = (w: number) => w === data.gw
    ? ({ saved: "As picked", carried: "Last week's team", locked: "As locked in" } as const)[team?.source ?? "saved"]
    : `The same 15${chip === "freehit" ? " (the squad before the Free Hit)" : ""}, with their best XI and captain for GW${w}`;

  const tiles: TileProps[] = [
    { label: `GW${data.gw} forecast`, value: `${int(first.xp)} xP`,
      note: team ? `captain doubled${chip === "3xc" ? " (tripled)" : ""}${chip === "bboost" ? ", bench included" : ""}` : "expected points, as the week is scored" },
    ...(first.points ? [
      { label: "Likely range", value: band(first.points.p10, first.points.p90),
        note: `the middle 80% of simulated scores; median ${Math.round(first.points.p50)}` },
      ...(first.target != null ? [{ label: `Chance of ${first.target}+`, value: pct(first.p_target), note: "in the simulated weeks" }] : []),
    ] : []),
    ...(later ? [{ label: `${span} forecast`, value: `${int(data.total.xp)} xP`,
      note: "every week added up, after any transfer hits" +
        (data.total.points ? `; likely ${band(data.total.points.p10, data.total.points.p90)}` : "") }] : []),
    ...(a ? [{ label: "The transfers", value: `${signed(a.xp, 0)} xP`,
      note: `over ${span}, after transfer hits, against keeping the GW${data.gw - 1} squad` +
        (a.p_better != null ? `; ahead in ${pct(a.p_better)} of simulated weeks` : "") }] : []),
  ];

  const weekColumns: Column<MyWeekNumbers>[] = [
    { key: "gw", label: "GW", numeric: true, value: (w) => w.gw },
    { key: "xp", label: "Forecast", numeric: true, value: (w) => w.xp, render: (w) => <strong>{pts(w.xp)}</strong>,
      title: "xP, captain doubled, before transfer hits" },
    { key: "range", label: "Likely range", numeric: true, value: (w) => w.points?.p90 ?? null,
      render: (w) => (w.points ? band(w.points.p10, w.points.p90) : "–"), title: "The middle 80% of simulated scores" },
    { key: "median", label: "Median", numeric: true, value: (w) => w.points?.p50 ?? null,
      render: (w) => (w.points ? Math.round(w.points.p50) : "–") },
    ...(team ? [
      { key: "captain", label: "Captain", value: (w: MyWeekNumbers) => name(team.weeks.find((t) => t.gw === w.gw)?.captain ?? 0) },
      { key: "bench", label: "Bench xP", numeric: true, value: (w: MyWeekNumbers) => team.weeks.find((t) => t.gw === w.gw)?.bench_xp ?? null,
        render: (w: MyWeekNumbers) => pts(team.weeks.find((t) => t.gw === w.gw)?.bench_xp) },
    ] as Column<MyWeekNumbers>[] : []),
    { key: "team", label: "Team", value: (w) => (w.gw === data.gw ? 0 : 1),
      render: (w) => (w.gw === data.gw ? described(w.gw) : <span className="muted">Same 15, best XI</span>) },
  ];

  const squadColumns: Column<SquadRow>[] = [
    { key: "name", label: "Player", value: (r) => r.player.web_name },
    { key: "club", label: "Club", value: (r) => site.team.get(r.player.team)?.short, render: (r) => <Club id={r.player.team} /> },
    { key: "pos", label: "Pos", value: (r) => r.player.element_type, render: (r) => POSITIONS[r.player.element_type] },
    { key: "role", label: `GW${data.gw}`, value: (r) => r.order, render: (r) => r.role, title: `His place in the GW${data.gw} team` },
    ...data.gameweeks.map((g): Column<SquadRow> => ({ key: `xp${g}`, label: `GW${g}`, group: "xP", numeric: true,
                                                      value: (r) => r[`xp_${g}`], render: (r) => pts(r[`xp_${g}`]) })),
    { key: "total", label: "Total", group: "xP", numeric: true, value: (r) => r.xp_total, render: (r) => <strong>{pts(r.xp_total)}</strong> },
    ...(squad.some((r) => r.pts_p90 != null) ? [
      { key: "range", label: "Range", group: `GW${data.gw} simulated`, numeric: true, value: (r: SquadRow) => r.pts_p90,
        render: (r: SquadRow) => band(r.pts_p10, r.pts_p90), title: `The middle 80% of his simulated GW${data.gw} points` } as Column<SquadRow>,
      { key: "haul", label: "10+", group: `GW${data.gw} simulated`, numeric: true, value: (r: SquadRow) => r.p_haul,
        render: (r: SquadRow) => pct(r.p_haul) } as Column<SquadRow>,
      { key: "blank", label: "≤2", group: `GW${data.gw} simulated`, numeric: true, value: (r: SquadRow) => r.p_blank,
        render: (r: SquadRow) => pct(r.p_blank), title: "2 points or fewer, not playing included" } as Column<SquadRow>,
    ] : []),
    ...(squad.some((r) => r.p_play != null) ? [{ key: "play", label: "Plays", numeric: true, value: (r: SquadRow) => r.p_play,
      render: (r: SquadRow) => pct(r.p_play), title: `This model's chance he plays in GW${data.gw}` } as Column<SquadRow>] : []),
  ];

  return (
    <>
      <h2>My Team: GW{data.gw}</h2>
      <p className="lede">
        {data.locked
          ? <>The team I'm playing in GW{data.gw} (its deadline was {when(data.deadline)}) and how it could do
              {later ? ` this week and the ${data.gameweeks.length - 1} after` : ""}, from the forecast made before the deadline
              {data.sims ? ` and ${data.sims.toLocaleString()} simulated gameweeks` : ""}. Once the gameweek is finished this page
              moves on to GW{data.gw + 1}, and <a href="#gameweeks">Past Gameweeks</a> shows how it went.</>
          : <>The team I'm planning to play in GW{data.gw}, whose deadline is {when(data.deadline)}, and how it could do
              {later ? ` this week and the ${data.gameweeks.length - 1} after` : ""}
              {data.sims ? `, from ${data.sims.toLocaleString()} simulated gameweeks` : ""}. It changes whenever I pick a new
              team before the deadline, and stays until the gameweek is finished.</>}
      </p>
      {team && (
        <div className="card">
          <p style={{ marginTop: 0 }}>
            {team.source === "saved" && <>Picked {when(team.saved_at)}.</>}
            {team.source === "locked" && <>Locked in at the deadline.</>}
            {team.source === "carried" && <>No new team picked for GW{data.gw} yet, so this is the GW{data.gw - 1} team carried
              over: the same XI, captain and bench, no transfers.</>}
            {team.source !== "carried" && !team.transfers.length && " No transfers."}
            {chip && <> <span className="tag">{CHIP_NAMES[chip] ?? chip}</span></>}
            {unlocked && <> <button className="link" onClick={lock}>Hide the team again</button></>}
          </p>
          {team.transfers.length > 0 && <Transfers moves={team.transfers} hits={team.hits} />}
          {team.settings && <Settings settings={team.settings} locked={team.source === "locked"} />}
        </div>
      )}
      <Tiles tiles={tiles} />

      {later && (
        <div className="toolbar">
          <Segmented label="Gameweek" value={shown} onChange={setChosen}
                     options={data.gameweeks.map((g) => ({ value: g, label: `GW${g}` }))} />
          {team && <span className="muted">{described(shown)}</span>}
        </div>
      )}

      {team && week ? (
        <div className="card">
          <Pitch gw={week.gw} lineup={week.lineup} bench={week.bench} captain={week.captain} vice={week.vice} flags
                 line={(p) => {
                   const r = byId.get(p);
                   return <><b>{pts(r?.[`xp_${week.gw}`])}</b> xP<span className="xp-extra">{week.gw === data.gw && r?.pts_p90 != null
                     ? ` · ${band(r.pts_p10, r.pts_p90)}` : ` · ${money(site.player.get(p)?.now_cost)}`}</span></>;
                 }} />
          <p className="note">
            xP is each player's expected points for GW{week.gw}{week.gw === data.gw ? ", then the middle 80% of his simulated points" : ""}.
            The fixture is shaded by FPL's difficulty rating, and a number in the corner is FPL's chance of playing.
            {later && " Later transfers aren't picked yet, so later weeks keep the same 15."}
          </p>
        </div>
      ) : !data.locked && (
        data.sealed
          ? <Unlock gw={data.gw} busy={state === "busy"} wrong={state === "wrong"} onUnlock={(word) => void unlock(word, true)} />
          : <div className="card"><p style={{ margin: 0 }}>The players, captain, chip and transfers are hidden until the GW{data.gw} deadline.</p></div>
      )}

      {numbers.points && (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>How GW{numbers.gw} could go</h3>
          <p style={{ marginTop: 0 }}>
            In the middle 80% of simulated weeks the team scores <strong>{band(numbers.points.p10, numbers.points.p90)}</strong> (median{" "}
            {Math.round(numbers.points.p50)}, average {pts(numbers.points.mean)}: a little above the xP because the simulation plays the
            auto-subs and the vice-captain).
          </p>
          <ScoreChart spread={numbers.points} forecast={numbers.xp} gw={numbers.gw} />
          {later && data.total.points && <>
            <h4>{span} in total</h4>
            <ScoreChart spread={data.total.points} forecast={data.total.xp} gw={data.gw}
                        label={`Simulated ${span} points, after transfer hits`} />
          </>}
        </div>
      )}

      {later && <>
        <h3>Week by week</h3>
        <Table columns={weekColumns} data={data.weeks} rowKey={(w) => w.gw} cardTitle={(w) => `GW${w.gw}`} cardStats={["xp", "range"]} />
      </>}

      {team && team.captains.length > 0 && data.sims && first.points && <>
        <h3>Captain options, GW{data.gw}</h3>
        <CaptainOddsTable sim={{ sims: data.sims, points: first.points, captains: team.captains, chips: {} }} captain={team.weeks[0].captain} />
        <p className="note">
          Each option's own points in the simulations, before the armband. Best pick: how often he outscores every other option
          in the same simulated week.
        </p>
      </>}

      {a && (
        <p>
          <strong>The transfers</strong> are worth {signed(a.xp, 1)} xP over {span} after transfer hits, against keeping the GW{data.gw - 1} squad
          with its best XI and captain each week.
          {a.p_better != null && <> In the same simulated weeks (the same goals, injuries and clean sheets for both) the new team is ahead
            in {pct(a.p_better)} of them, by {signed(a.p10 ?? 0, 0)} to {signed(a.p90 ?? 0, 0)} points in the middle 80%.</>}
        </p>
      )}

      {team && <>
        <h3>The squad</h3>
        <Table columns={squadColumns} data={squad} sort="role" desc={false} rowKey={(r) => r.element}
               cardSub={["club", "pos", "role"]} cardStats={[`xp${data.gw}`, "total"]} />
      </>}

      <Note>
        xP from <strong>{data.model}</strong>, the forecast saved for GW{data.gw}. Every simulated week draws the goals in each
        fixture, who plays, goals, assists, clean sheets, bonus and cards, with each player's average matched to his xP, then scores
        the team as FPL does. See <a href="#about">About</a> for how.
      </Note>
    </>
  );
}
