import { useMemo, useState } from "react";
import { Club, Loading, Note, Segmented, Sheet, Stats, Table, Tiles, usePhone, type Column } from "../components/ui";
import { rows, type Forecast, type ModelTeam, type NextGw } from "../data";
import { POSITIONS, STATUS, money, pct, pts, when } from "../format";
import { NEWS_RULE, ago, newsUrl, shortDay, type Headline, type NewsChange, type NewsLog, type NewsPlayer, type PressClub } from "../news";
import { useData, useSite } from "../site";

type View = "out" | "doubt" | "all";
type Since = 0 | 2 | 7;

const ELSEWHERE = [
  { label: "Fantasy Football Scout: injuries and bans", url: "https://www.fantasyfootballscout.co.uk/fantasy-football-injuries" },
];

/** A list of headlines: each links out to the article (through NewsNow), with its publisher and how long ago. */
function Headlines({ items }: { items: Headline[] }) {
  const site = useSite();
  return (
    <ul className="headlines">
      {items.map((h) => (
        <li key={h.id}>
          <a href={h.url} target="_blank" rel="noreferrer nofollow">{h.title}</a>
          <span className="headline-meta">
            {h.publisher}{h.t && <> · <span title={when(h.t)}>{ago(h.t)}</span></>}
            {h.players.length > 0 && <> · {h.players.map((id, i) => (
              <span key={id}>{i > 0 && ", "}<a href={`#news/${id}`}>{site.player.get(id)?.web_name ?? id}</a></span>))}</>}
          </span>
        </li>
      ))}
    </ul>
  );
}

/** FPL's status as a coloured word: out (injured, suspended, left) red, doubtful amber-ish. */
function Status({ p }: { p: NewsPlayer }) {
  const word = STATUS[p.status] ?? "Available";
  const cls = p.status === "d" ? "tag" : p.status === "a" ? "tag" : "tag warn";
  return <span className={cls}>{word}{p.chance !== null && p.status !== "u" ? ` ${p.chance}%` : ""}</span>;
}

function PressTag({ said }: { said: "OUT" | "DOUBT" | "IN" | null }) {
  if (!said) return <span className="muted">–</span>;
  return <span className={said === "OUT" ? "tag warn" : said === "IN" ? "tag good-tag" : "tag"}>{said}</span>;
}

/** Everything known about one player's news: FPL's text, its source and time, the return date, the press
 * conference and the chance of playing this model used. */
function Detail({ p, log, forecast, club }: { p: NewsPlayer; log: NewsLog; forecast?: Forecast; club?: PressClub }) {
  const history = log.log.filter((c) => c.id === p.id).sort((a, b) => (b.t ?? "").localeCompare(a.t ?? ""));
  const said = club?.players.find((x) => x.id === p.id);
  const lines = (log.headlines?.items ?? []).filter((h) => h.players.includes(p.id));
  return (
    <div>
      {p.news ? <p style={{ fontSize: 15 }}>{p.news}</p> : <p className="muted">No news from FPL.</p>}
      <Stats items={[
        { label: "FPL status", value: <Status p={p} /> },
        { label: "Expected back", value: p.back ? shortDay(p.back) : "–", title: "FPL's date: available for matches from that day" },
        { label: "Updated", value: p.added ? <>{when(p.added)} <span className="muted">({ago(p.added)})</span></> : "–",
          title: "When FPL last changed this player's news" },
        { label: "Press conference", value: <PressTag said={p.press} /> },
        ...(forecast?.avail != null ? [{ label: `Chance used, GW${log.gw_next}`, value: pct(forecast.avail),
                                         title: `Set by ${NEWS_RULE[forecast.news_rule ?? "flag"]}` }] : []),
        ...(forecast ? [{ label: `xP GW${log.gw_next}`, value: pts(forecast[`xp_${log.gw_next ?? 0}`]) }] : []),
      ]} />
      {p.source && <p><a className="link" href={p.source} target="_blank" rel="noreferrer">Source article ↗</a></p>}
      {p.risks.length > 0 && (
        <p className="note">Known absences ahead: {p.risks.map((r, i) => <span key={i}>{i > 0 && " · "}GW{r.gw}: {r.notes}</span>)}</p>
      )}
      {club && (said || club.quotes.length > 0) && (
        <>
          <h4 style={{ margin: "12px 0 4px" }}>{club.club}'s press conference, {shortDay(log.press?.updated)}</h4>
          {!log.press?.fresh && <p className="note" style={{ marginTop: 0 }}>Before the last deadline: not used for Gameweek {log.gw_next}.</p>}
          {said && <p className="press-line">About him: <PressTag said={said.status} /></p>}
          {club.quotes.map((q, i) => <Quote key={i} text={q.text} by={q.by} />)}
        </>
      )}
      {lines.length > 0 && (
        <>
          <h4 style={{ margin: "12px 0 4px" }}>In the headlines</h4>
          <Headlines items={lines} />
        </>
      )}
      {history.length > 1 && (
        <>
          <h4 style={{ margin: "12px 0 4px" }}>His news this season</h4>
          <ul className="news-history">
            {history.map((c, i) => (
              <li key={i}><span className="muted">{c.t ? shortDay(c.t) : `GW${c.gw}`}</span> {c.news || "Back to full fitness"}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

function Quote({ text, by }: { text: string; by: string | null }) {
  return (
    <blockquote className="quote">
      “{text}”{by && <div className="quote-by">— {by}</div>}
    </blockquote>
  );
}

function PressCard({ club }: { club: PressClub }) {
  return (
    <div className="card press-card">
      <div className="press-head">{club.team ? <Club id={club.team} /> : null} <strong>{club.club}</strong></div>
      {(["OUT", "DOUBT", "IN"] as const).map((said) => {
        const names = club.players.filter((p) => p.status === said);
        return names.length ? (
          <p key={said} className="press-line">
            <PressTag said={said} />{" "}
            {names.map((p, i) => (
              <span key={i}>{i > 0 && ", "}{p.id ? <a href={`#news/${p.id}`}>{p.name}</a> : p.name}</span>
            ))}
          </p>
        ) : null;
      })}
      {club.quotes.map((q, i) => <Quote key={i} text={q.text} by={q.by} />)}
    </div>
  );
}

export default function News() {
  const site = useSite();
  const live = useData<NewsLog>(newsUrl());
  const local = useData<NewsLog>(live === null ? "news.json" : null);
  const log = live ?? local;
  const next = useData<NextGw>("next.json");
  const modelTeam = useData<ModelTeam>("modelteam.json");
  const [view, setView] = useState<View>("out");
  const [pos, setPos] = useState(0);
  const [club, setClub] = useState(0);
  const [since, setSince] = useState<Since>(0);
  const [search, setSearch] = useState("");
  const [modelOnly, setModelOnly] = useState(false);
  const [headlineClub, setHeadlineClub] = useState(0);
  const selectedId = Number(window.location.hash.split("/")[1]) || null;
  const phone = usePhone();

  const forecasts = useMemo(() => new Map(rows<Forecast>(next?.players).map((f) => [f.element, f])), [next]);
  const modelSquad = useMemo(() => {
    const weeks = modelTeam?.gameweeks ?? [];
    const latest = modelTeam?.next ?? weeks[weeks.length - 1];
    return new Set(latest?.squad ?? []);
  }, [modelTeam]);
  const pressByTeam = useMemo(() => new Map((log?.press?.clubs ?? []).map((c) => [c.team, c])), [log]);

  const list = useMemo(() => {
    if (!log) return [];
    const q = search.trim().toLowerCase();
    const now = Date.now();
    return log.players.filter((p) =>
      (view === "all" || (view === "out" ? ["i", "s", "n"].includes(p.status) : p.status === "d")) &&
      (!pos || p.pos === pos) && (!club || p.team === club) && (!modelOnly || modelSquad.has(p.id)) &&
      (!since || (p.added !== null && now - Date.parse(p.added) <= since * 86_400_000)) &&
      (!q || p.name.toLowerCase().includes(q) || (p.second ?? "").toLowerCase().includes(q)));
  }, [log, view, pos, club, since, search, modelOnly, modelSquad]);

  const changes = useMemo(() => {
    const cutoff = Date.now() - 14 * 86_400_000;
    return (log?.log ?? []).filter((c) => c.t && Date.parse(c.t) >= cutoff)
      .map((c, i) => ({ ...c, key: i })).sort((a, b) => (b.t ?? "").localeCompare(a.t ?? ""));
  }, [log]);

  const headlineItems = log?.headlines?.items ?? [];
  const shownHeadlines = headlineClub
    ? headlineItems.filter((h) => h.teams.includes(headlineClub) || h.players.some((id) => site.player.get(id)?.team === headlineClub))
    : headlineItems;

  if (log === undefined || (log === null && local === undefined)) return <Loading />;
  if (!log) return <p>The team news hasn't been collected yet.</p>;

  const press = log.press;
  const selected = selectedId ? log.players.find((p) => p.id === selectedId) : undefined;
  const xpNext = (id: number) => forecasts.get(id)?.[`xp_${log.gw_next ?? 0}`] ?? null;
  const detail = (p: NewsPlayer) => <Detail p={p} log={log} forecast={forecasts.get(p.id)} club={pressByTeam.get(p.team)} />;
  const columns: Column<NewsPlayer>[] = [
    { key: "name", label: "Player", value: (p) => p.name, render: (p) => (
      <>{p.name}{modelSquad.has(p.id) && <> <span className="tag" title="In this model's team">MT</span></>}</>) },
    { key: "team", label: "Club", value: (p) => site.team.get(p.team)?.short, render: (p) => <Club id={p.team} /> },
    { key: "pos", label: "Pos", value: (p) => p.pos, render: (p) => POSITIONS[p.pos] },
    { key: "status", label: "Status", sortable: true, value: (p) => p.chance ?? (p.status === "a" ? 100 : 0), render: (p) => <Status p={p} />,
      title: "FPL's status and chance of playing next round" },
    { key: "reason", label: "Reason", value: (p) => p.reason, wrap: true },
    { key: "back", label: "Back", short: "Back", sortable: true, value: (p) => p.back ?? "", render: (p) => (p.back ? shortDay(p.back) : "–"),
      title: "FPL's expected return date" },
    { key: "press", label: "Press", value: (p) => p.press ?? "", render: (p) => <PressTag said={p.press} />,
      title: "What the manager said at his press conference before this gameweek" },
    { key: "used", label: "Chance used", numeric: true, value: (p) => forecasts.get(p.id)?.avail ?? null,
      render: (p) => pct(forecasts.get(p.id)?.avail), title: "The chance of playing this model used for the next gameweek" },
    { key: "updated", label: "Updated", sortable: true, value: (p) => p.added ?? "", render: (p) => (p.added ? ago(p.added) : "–"),
      title: "When FPL last changed this player's news" },
    { key: "xp", label: log.gw_next ? `xP GW${log.gw_next}` : "xP next", numeric: true, value: (p) => xpNext(p.id), render: (p) => pts(xpNext(p.id)) },
    { key: "sel", label: "Sel %", numeric: true, value: (p) => p.selected, render: (p) => p.selected.toFixed(1) },
    { key: "source", label: "Source", value: (p) => p.source ?? "",
      render: (p) => (p.source ? <a className="link" href={p.source} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>Article ↗</a> : "–") },
  ];
  const changeColumns: Column<NewsChange & { key: number }>[] = [
    { key: "t", label: "When", sortable: true, value: (c) => c.t ?? "", render: (c) => when(c.t) },
    { key: "name", label: "Player", value: (c) => c.name, render: (c) => <a href={`#news/${c.id}`}>{c.name}</a> },
    { key: "team", label: "Club", value: (c) => site.teamByCode.get(c.team_code)?.short, render: (c) => <Club code={c.team_code} /> },
    { key: "status", label: "Status", value: (c) => STATUS[c.status] ?? "Available",
      render: (c) => <span className={c.status === "a" ? "good" : c.status === "d" ? undefined : "bad"}>{STATUS[c.status] ?? "Available"}{c.chance !== null && c.status !== "u" ? ` ${c.chance}%` : ""}</span> },
    { key: "news", label: "News", value: (c) => c.news || "Back to full fitness", wrap: true },
  ];

  const out = log.players.filter((p) => ["i", "s", "n"].includes(p.status)).length;
  const doubts = log.players.filter((p) => p.status === "d").length;
  const recent = log.players.filter((p) => p.added && Date.now() - Date.parse(p.added) <= 2 * 86_400_000).length;
  return (
    <>
      <h2>Team News</h2>
      <p className="lede">Who's injured, suspended or doubtful, when they're expected back and where the news came from:
        FPL's own news for every player, timed to when FPL last changed it, with a link to the article behind it, and the
        managers' words from their press conferences. Click a player for everything known about him and the chance of
        playing this model gave him.</p>
      <Tiles tiles={[
        { label: "Injured or suspended", value: out },
        { label: "Doubtful", value: doubts },
        { label: "Changed in the last 48 h", value: recent },
        { label: "Press conferences", value: press?.fresh ? `GW${log.gw_next}` : "Not yet",
          note: press?.updated ? `updated ${shortDay(press.updated)}` : undefined },
        { label: "Last checked", value: <span style={{ fontSize: 17 }}>{when(log.updated)}</span>, note: live ? "every hour" : "when the site was published" },
      ]} />
      {selected && !phone && (
        <div className="card">
          <div className="toolbar" style={{ justifyContent: "space-between" }}>
            <div>
              <strong style={{ fontSize: 17 }}>{selected.name}</strong> <Club id={selected.team} />{" "}
              <span className="muted">{POSITIONS[selected.pos]} · {money(selected.cost / 10)}</span>
            </div>
            <a href="#news" className="link">Close</a>
          </div>
          {detail(selected)}
        </div>
      )}
      {phone && (
        <Sheet open={!!selected} onClose={() => { window.location.hash = "news"; }}
               title={selected && <>{selected.name} <Club id={selected.team} /> <span className="muted">{POSITIONS[selected.pos]}</span></>}>
          {selected && detail(selected)}
        </Sheet>
      )}
      {selectedId && !selected && (() => {
        const fit = site.player.get(selectedId);
        const lines = headlineItems.filter((h) => h.players.includes(selectedId));
        return (
          <div className="card">
            <div className="toolbar" style={{ justifyContent: "space-between" }}>
              <div>
                {fit && <><strong style={{ fontSize: 17 }}>{fit.web_name}</strong> <Club id={fit.team} /></>}{" "}
                <span className="muted">No FPL news: fit, as far as FPL knows.</span>
              </div>
              <a href="#news" className="link">Close</a>
            </div>
            {lines.length > 0 && <><h4 style={{ margin: "12px 0 4px" }}>In the headlines</h4><Headlines items={lines} /></>}
          </div>
        );
      })()}

      <div className="toolbar" style={{ marginTop: 14 }}>
        <Segmented label="Status" value={view} onChange={setView}
                   options={[{ value: "out", label: "Out" }, { value: "doubt", label: "Doubtful" }, { value: "all", label: "All" }]} />
        <Segmented label="Position" value={pos} onChange={setPos}
                   options={[{ value: 0, label: "All" }, ...Object.entries(POSITIONS).map(([k, v]) => ({ value: Number(k), label: v }))]} />
        <Segmented label="Changed" value={since} onChange={setSince}
                   options={[{ value: 0, label: "Any time" }, { value: 2, label: "48 h" }, { value: 7, label: "7 days" }]} />
        <select aria-label="Club" value={club} onChange={(e) => setClub(Number(e.target.value))}>
          <option value={0}>All clubs</option>
          {[...site.meta.teams].sort((a, b) => a.name.localeCompare(b.name)).map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
        <input type="search" placeholder="Search name" aria-label="Search players" value={search} onChange={(e) => setSearch(e.target.value)} />
        <label><input type="checkbox" checked={modelOnly} onChange={(e) => setModelOnly(e.target.checked)} /> This model's team only</label>
      </div>
      <Table key={view} columns={columns} data={list} sort="updated" desc rowKey={(p) => p.id} limit={40}
             selected={selectedId} cardSub={["team", "pos", "reason"]} cardStats={["status", "back"]}
             onRow={(p) => { window.location.hash = `news/${p.id}`; if (!phone) window.scrollTo({ top: 0, behavior: "smooth" }); }} />
      <Note>
        "Back" is FPL's expected return date: he should be available for matches from that day. For the next gameweek
        this model uses FPL's chance of playing, unless a press conference given after FPL's last update says otherwise
        (out, doubtful or fit again); for the weeks after, the return date (none before it, then a 75% chance for an
        injury and full availability after a ban) and known absences such as a loanee facing his parent club.
      </Note>

      <h3>Press conferences</h3>
      {!press || press.clubs.length === 0 ? <Note>The press-conference summaries couldn't be read.</Note> : (
        <>
          <p className="note">
            {press.fresh
              ? <>The managers' team news before Gameweek {log.gw_next}, summarised by </>
              : <>Last updated {shortDay(press.updated)}, before the last deadline: the summaries for Gameweek {log.gw_next} aren't
                  out yet, so this model isn't using these. Summarised by </>}
            <a className="link" href={log.press_url} target="_blank" rel="noreferrer">{log.press_name} ↗</a>.
          </p>
          <div className="press-grid">
            {[...press.clubs].sort((a, b) => a.club.localeCompare(b.club)).map((c) => <PressCard key={c.club} club={c} />)}
          </div>
        </>
      )}

      <h3>Latest headlines</h3>
      {headlineItems.length === 0 ? <Note>The headlines couldn't be read.</Note> : (
        <>
          <div className="toolbar">
            <select aria-label="Club" value={headlineClub} onChange={(e) => setHeadlineClub(Number(e.target.value))}>
              <option value={0}>All clubs</option>
              {[...site.meta.teams].sort((a, b) => a.name.localeCompare(b.name)).map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
            </select>
            <span className="muted">{shownHeadlines.length} headlines</span>
          </div>
          <Headlines items={shownHeadlines.slice(0, 40)} />
          <p className="note">
            The last two weeks from <a className="link" href={log.headlines_url} target="_blank" rel="noreferrer">{log.headlines_name} ↗</a>'s
            Premier League injuries and suspensions page, checked every few hours (hourly before a deadline). Players and clubs are
            picked out by name, so check the odd one. They're here to read: this model doesn't use them.
          </p>
        </>
      )}

      <h3>Changes to FPL's news, last 14 days</h3>
      <Table columns={changeColumns} data={changes} sort="t" desc rowKey={(c) => c.key} limit={30}
             cardSub={["team", "status"]} cardStats={["t"]} />

      <h3>Elsewhere</h3>
      <ul>
        {ELSEWHERE.map((e) => <li key={e.url}><a className="link" href={e.url} target="_blank" rel="noreferrer">{e.label} ↗</a></li>)}
      </ul>
      <Note>
        FPL's news is the Premier League's own (the same text, flag and source link as in the FPL app), checked every hour.
        Headlines come from NewsNow, credited to each publisher.
        Fantasy Football Scout's table adds return dates and sources for some players, but its terms don't allow copying
        it automatically, so it's linked rather than shown here.
      </Note>
    </>
  );
}
