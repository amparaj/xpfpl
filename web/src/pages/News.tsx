import { useEffect, useMemo } from "react";
import { Club, Loading, Note, Tiles } from "../components/ui";
import { Headlines, PressSummary, playerHref, useNewsLog } from "../components/PlayerNews";
import { when } from "../format";
import { shortDay, type Headline, type NewsLog, type NewsPlayer } from "../news";
import { useHash, useSite } from "../site";

// Team News, club by club: FPL's flags, the manager's press conference and the headlines for each club.
// Each player's own news (the full text, source, return date, history) is on his Players page.

const ELSEWHERE = [
  { label: "Fantasy Football Scout: injuries and bans", url: "https://www.fantasyfootballscout.co.uk/fantasy-football-injuries" },
];
const HEADLINES_PER_CLUB = 3;

const isOut = (p: NewsPlayer) => ["i", "s", "n"].includes(p.status);

/** A club's flagged players under one tag: names link to their Players page, with the return date or chance. */
function Flagged({ players, out }: { players: NewsPlayer[]; out: boolean }) {
  if (!players.length) return null;
  return (
    <p className="press-line">
      <span className={out ? "tag warn" : "tag"}>{out ? "OUT" : "DOUBT"}</span>{" "}
      {players.map((p, i) => (
        <span key={p.id}>
          {i > 0 && ", "}<a href={playerHref(p.id)} title={p.news}>{p.name}</a>
          <span className="muted">
            {out ? (p.back ? ` (back ${shortDay(p.back)})` : p.status === "s" ? " (banned)" : "")
                 : p.chance !== null ? ` (${p.chance}%)` : ""}
          </span>
        </span>
      ))}
    </p>
  );
}

function ClubCard({ team, log, headlines }: { team: { id: number; name: string }; log: NewsLog; headlines: Headline[] }) {
  const flagged = log.players.filter((p) => p.team === team.id && p.status !== "a")
    .sort((a, b) => (a.chance ?? 0) - (b.chance ?? 0) || a.name.localeCompare(b.name));
  const out = flagged.filter(isOut), doubts = flagged.filter((p) => p.status === "d");
  const press = log.press?.clubs.find((c) => c.team === team.id);
  return (
    <div className="card press-card" id={`club-${team.id}`}>
      <div className="press-head">
        <Club id={team.id} /> <strong>{team.name}</strong>
        <span className="muted" style={{ marginLeft: "auto", fontSize: 13 }}>
          {out.length} out · {doubts.length} doubtful
        </span>
      </div>
      <h4 className="club-news-h">FPL's flags</h4>
      {flagged.length ? <><Flagged players={out} out /><Flagged players={doubts} out={false} /></>
        : <p className="muted press-line">Nobody flagged.</p>}
      <h4 className="club-news-h">Press conference</h4>
      {press && (press.players.length || press.quotes.length)
        ? <PressSummary club={press} /> : <p className="muted press-line">No summary yet.</p>}
      {headlines.length > 0 && (
        <>
          <h4 className="club-news-h">Headlines</h4>
          <Headlines items={headlines.slice(0, HEADLINES_PER_CLUB)} />
          {headlines.length > HEADLINES_PER_CLUB && (
            <details>
              <summary className="link">{headlines.length - HEADLINES_PER_CLUB} more</summary>
              <Headlines items={headlines.slice(HEADLINES_PER_CLUB, 20)} />
            </details>
          )}
        </>
      )}
    </div>
  );
}

export default function News() {
  const site = useSite();
  const log = useNewsLog();
  const hash = useHash();
  const parts = hash.split("/");
  // Old links went to #news/<player>: a player's news is on his Players page now.
  useEffect(() => {
    if (parts[1] && parts[1] !== "club") window.location.replace(`#players/${parts[1]}`);
  }, [hash]); // eslint-disable-line react-hooks/exhaustive-deps
  const club = parts[1] === "club" ? Number(parts[2]) || 0 : 0;
  const setClub = (id: number) => { window.location.hash = id ? `news/club/${id}` : "news"; };

  const byClub = useMemo(() => {
    const out = new Map<number, Headline[]>();
    for (const h of log?.headlines?.items ?? []) {
      const teams = new Set([...h.teams, ...h.players.map((id) => site.player.get(id)?.team).filter((t): t is number => !!t)]);
      for (const t of teams) out.set(t, [...(out.get(t) ?? []), h]);
    }
    return out;
  }, [log, site]);

  if (log === undefined) return <Loading />;
  if (!log) return <p>The team news hasn't been collected yet.</p>;

  const press = log.press;
  const teams = [...site.meta.teams].sort((a, b) => a.name.localeCompare(b.name));
  const shown = club ? teams.filter((t) => t.id === club) : teams;
  const out = log.players.filter(isOut).length;
  const doubts = log.players.filter((p) => p.status === "d").length;
  const recent = log.players.filter((p) => p.added && Date.now() - Date.parse(p.added) <= 2 * 86_400_000).length;
  return (
    <>
      <h2>Team News</h2>
      <p className="lede">Each club's injuries, suspensions and doubts: FPL's own flags, what the manager said at his press
        conference, and the latest headlines. Click a player for his full news (where it came from, when he's expected back,
        his news this season and the chance of playing this model gave him) on his <a href="#players">Players</a> page, where
        you can also list everyone who's out or doubtful.</p>
      <Tiles tiles={[
        { label: "Injured or suspended", value: out },
        { label: "Doubtful", value: doubts },
        { label: "Changed in the last 48 h", value: recent },
        { label: "Press conferences", value: press?.fresh ? `GW${log.gw_next}` : "Not yet",
          note: press?.updated ? `updated ${shortDay(press.updated)}` : undefined },
        { label: "Last checked", value: <span style={{ fontSize: 17 }}>{when(log.updated)}</span>, note: "every hour" },
      ]} />

      <div className="toolbar" style={{ marginTop: 14 }}>
        <select aria-label="Club" value={club} onChange={(e) => setClub(Number(e.target.value))}>
          <option value={0}>All clubs</option>
          {teams.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
        <a className="link" href="#players/out">Everyone who's out →</a>
      </div>
      {press && (
        <p className="note" style={{ marginBottom: 10 }}>
          {press.fresh
            ? <>Press conferences before Gameweek {log.gw_next}, summarised by </>
            : <>Press conferences last updated {shortDay(press.updated)}, before the last deadline: the summaries for
                Gameweek {log.gw_next} aren't out yet, so this model isn't using these. Summarised by </>}
          <a className="link" href={log.press_url} target="_blank" rel="noreferrer">{log.press_name} ↗</a>.
        </p>
      )}
      <div className="press-grid">
        {shown.map((t) => <ClubCard key={t.id} team={t} log={log} headlines={byClub.get(t.id) ?? []} />)}
      </div>

      <Note>
        For the next gameweek this model uses FPL's chance of playing, unless a press conference given after FPL's last
        update says otherwise (out, doubtful or fit again); for the weeks after, FPL's return date (none before it, then a
        75% chance for an injury and full availability after a ban) and known absences such as a loanee facing his parent club.
      </Note>

      <h3>Elsewhere</h3>
      <ul>
        {ELSEWHERE.map((e) => <li key={e.url}><a className="link" href={e.url} target="_blank" rel="noreferrer">{e.label} ↗</a></li>)}
      </ul>
      <Note>
        FPL's news is the Premier League's own (the same text, flag and source link as in the FPL app), checked every hour.
        Headlines come from {log.headlines_url
          ? <a className="link" href={log.headlines_url} target="_blank" rel="noreferrer">{log.headlines_name} ↗</a> : "NewsNow"}'s
        Premier League injuries page, credited to each publisher, and are picked out by name, so check the odd one; this
        model doesn't use them. Fantasy Football Scout's table adds return dates and sources for some players, but its terms
        don't allow copying it automatically, so it's linked rather than shown here.
      </Note>
    </>
  );
}
