// Team news pieces shared by Players (each player's news) and Team News (club by club).

import { Stats } from "./ui";
import type { Forecast } from "../data";
import { flagLabel, pct, pts, when } from "../format";
import { NEWS_RULE, ago, newsUrl, shortDay, type Headline, type NewsLog, type NewsPlayer, type PressClub } from "../news";
import { useData, useSite } from "../site";

/** The news log: the live copy on the `news` branch, else the one the export wrote. undefined while loading,
 * null if neither could be read. */
export function useNewsLog(): NewsLog | null | undefined {
  const live = useData<NewsLog>(newsUrl());
  const local = useData<NewsLog>(live === null ? "news.json" : null);
  if (live === undefined || (live === null && local === undefined)) return undefined;
  return live ?? local;
}

/** Players' pages are where each player's news lives. */
export const playerHref = (id: number) => `#players/${id}`;

/** A list of headlines: each links out to the article (through NewsNow), with its publisher and how long ago. */
export function Headlines({ items }: { items: Headline[] }) {
  const site = useSite();
  return (
    <ul className="headlines">
      {items.map((h) => (
        <li key={h.id}>
          <a href={h.url} target="_blank" rel="noreferrer nofollow">{h.title}</a>
          <span className="headline-meta">
            {h.publisher}{h.t && <> · <span title={when(h.t)}>{ago(h.t)}</span></>}
            {h.players.length > 0 && <> · {h.players.map((id, i) => (
              <span key={id}>{i > 0 && ", "}<a href={playerHref(id)}>{site.player.get(id)?.web_name ?? id}</a></span>))}</>}
          </span>
        </li>
      ))}
    </ul>
  );
}

/** FPL's status as a tag: out (injured, suspended, left) red, doubtful plain with its chance. */
export function NewsStatus({ status, chance }: { status: string; chance: number | null }) {
  return <span className={status === "d" || status === "a" ? "tag" : "tag warn"}>{flagLabel(status, chance)}</span>;
}

export function PressTag({ said }: { said: "OUT" | "DOUBT" | "IN" | null }) {
  if (!said) return <span className="muted">–</span>;
  return <span className={said === "OUT" ? "tag warn" : said === "IN" ? "tag good-tag" : "tag"}>{said}</span>;
}

export function Quote({ text, by }: { text: string; by: string | null }) {
  return (
    <blockquote className="quote">
      “{text}”{by && <div className="quote-by">— {by}</div>}
    </blockquote>
  );
}

/** Everything known about one player's news: FPL's text, its source and time, the return date, the press
 * conference, the chance of playing this model used, the headlines naming him and his news this season.
 * `p` is undefined for a player FPL has no news on (then only headlines, if any). */
export function NewsDetail({ id, p, log, forecast }: { id: number; p?: NewsPlayer; log: NewsLog; forecast?: Forecast }) {
  const history = log.log.filter((c) => c.id === id).sort((a, b) => (b.t ?? "").localeCompare(a.t ?? ""));
  const club = p ? log.press?.clubs.find((c) => c.team === p.team) : undefined;
  const said = club?.players.find((x) => x.id === id);
  const lines = (log.headlines?.items ?? []).filter((h) => h.players.includes(id));
  if (!p && lines.length === 0) return null;
  return (
    <div className="player-news">
      <h4 style={{ margin: "0 0 4px" }}>Team news</h4>
      {p && (
        <>
          {p.news ? <p style={{ fontSize: 15, margin: "4px 0" }}>{p.news}</p> : <p className="muted">No news from FPL.</p>}
          <Stats items={[
            { label: "FPL status", value: <NewsStatus status={p.status} chance={p.chance} /> },
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
        </>
      )}
      {!p && <p className="muted" style={{ margin: "4px 0" }}>No news from FPL: fit, as far as FPL knows.</p>}
      {club && (said || club.quotes.length > 0) && (
        <>
          <h4 style={{ margin: "12px 0 4px" }}>{club.club}'s press conference, {shortDay(log.press?.updated)}</h4>
          {!log.press?.fresh && <p className="note" style={{ marginTop: 0 }}>Before the last deadline: not used for Gameweek {log.gw_next}.</p>}
          {said && <p className="press-line">About him: <PressTag said={said.status} /></p>}
          {club.quotes.map((q, i) => <Quote key={i} text={q.text} by={q.by} />)}
          <p style={{ margin: "4px 0 0" }}><a className="link" href={`#news/club/${p!.team}`}>All {club.club} news →</a></p>
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

/** One club's press conference: who the manager said is out, doubtful or back, and his words. */
export function PressSummary({ club }: { club: PressClub }) {
  return (
    <>
      {(["OUT", "DOUBT", "IN"] as const).map((said) => {
        const names = club.players.filter((p) => p.status === said);
        return names.length ? (
          <p key={said} className="press-line">
            <PressTag said={said} />{" "}
            {names.map((p, i) => (
              <span key={i}>{i > 0 && ", "}{p.id ? <a href={playerHref(p.id)}>{p.name}</a> : p.name}</span>
            ))}
          </p>
        ) : null;
      })}
      {club.quotes.map((q, i) => <Quote key={i} text={q.text} by={q.by} />)}
    </>
  );
}
