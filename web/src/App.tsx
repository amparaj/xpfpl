import { Component, useEffect, useRef, useState, type ReactNode } from "react";
import { Loading, usePhone } from "./components/ui";
import { when } from "./format";
import About from "./pages/About";
import Accuracy from "./pages/Accuracy";
import DataPage from "./pages/Data";
import Gameweeks from "./pages/Gameweeks";
import MarketsPage from "./pages/Markets";
import ModelTeamPage from "./pages/ModelTeam";
import MyTeamPage from "./pages/MyTeam";
import News from "./pages/News";
import NextGameweek from "./pages/NextGameweek";
import Players from "./pages/Players";
import Prices from "./pages/Prices";
import { SiteContext, useHash, useSiteData, type Site } from "./site";

// About first (where the site opens), then how far to trust the model, the weeks played and the
// week ahead with this model's own team and the team I'm planning, research, and the raw data last.
const PAGES = [
  { id: "about", label: "About", short: "About", component: About },
  { id: "accuracy", label: "This Model's Accuracy", short: "Accuracy", component: Accuracy },
  { id: "gameweeks", label: "Past Gameweeks", short: "Past GWs", component: Gameweeks },
  { id: "next", label: "Next Gameweek", short: "Next GW", component: NextGameweek },
  { id: "model-team", label: "This Model's Team", short: "Model's Team", component: ModelTeamPage },
  { id: "my-team", label: "My Team", short: "My Team", component: MyTeamPage },
  { id: "players", label: "Players", short: "Players", component: Players },
  { id: "news", label: "Team News", short: "News", component: News },
  { id: "prices", label: "Prices", short: "Prices", component: Prices },
  { id: "markets", label: "Markets", short: "Markets", component: MarketsPage },
  { id: "data", label: "Data", short: "Data", component: DataPage },
] as const;

export default function App() {
  const site = useSiteData();
  const hash = useHash();
  // The page is the part of the hash before any "/": #gameweeks/6/53 -> gameweeks.
  const page = PAGES.find((p) => p.id === hash.split("/")[0]) ?? PAGES[0];
  const Page = page.component;

  useEffect(() => {
    document.title = `${page.label} · xP-FPL: Expected Points for Fantasy Premier League`;
  }, [page]);

  // On a phone the header scrolls away but the page tabs stay pinned: the header sticks at minus
  // the height above the tabs. On a wide screen the whole header stays. The phone tabs wrap onto
  // a few lines with short labels, so every page is in view without scrolling sideways.
  const phone = usePhone();
  const header = useRef<HTMLElement>(null);
  const nav = useRef<HTMLElement>(null);
  const [offset, setOffset] = useState(0);
  useEffect(() => {
    const h = header.current, n = nav.current;
    if (!h || !n || !phone) { setOffset(0); return; }
    const measure = () => setOffset(n.offsetTop);
    const observer = new ResizeObserver(measure);
    observer.observe(h);
    measure();
    return () => observer.disconnect();
  }, [phone, site]);
  // Keep the current page's tab in view in the scrolling tab row.
  useEffect(() => {
    nav.current?.querySelector<HTMLElement>("a.on")?.scrollIntoView({ block: "nearest", inline: "center" });
  }, [page]);

  return (
    <>
      <header className="top" ref={header} style={offset ? { top: -offset } : undefined}>
        <div className="top-inner">
          <div className="brand">
            <h1>
              <a href="#about" className="brand-logo" aria-label="xP-FPL: About">
                <img src={`${import.meta.env.BASE_URL}logo-mark.png`} alt="" width={40} height={40} />
              </a>
              <span>xP-FPL<span className="brand-sub">: Expected Points for Fantasy Premier League</span></span>
            </h1>
            <span className="byline">Created by Ayush Parajuli</span>
          </div>
          {site && <Status site={site} />}
          <nav aria-label="Pages" ref={nav}>
            {PAGES.map((p) => (
              <a key={p.id} href={`#${p.id}`} className={p === page ? "on" : undefined}
                 aria-current={p === page ? "page" : undefined}>
                {phone ? p.short : p.label}
              </a>
            ))}
          </nav>
        </div>
      </header>
      <main>
        {site === undefined && <Loading />}
        {site === null && (
          <p>
            No data yet. Run <code>xpfpl export</code> to write <code>web/public/data/</code>.
          </p>
        )}
        {site && (
          <SiteContext.Provider value={site}>
            <Boundary key={page.id}>
              <Page />
            </Boundary>
          </SiteContext.Provider>
        )}
      </main>
      <footer>
        Expected points (xP) from a model trained on every Premier League season since 2016-17. Betting odds
        from Polymarket and Football-Data.co.uk. Not affiliated with the Premier League or Fantasy Premier League.
      </footer>
    </>
  );
}

/** Keeps one page's failure (a bad chart, an odd API response) from blanking the whole site. */
class Boundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    return this.state.error ? (
      <p>Something went wrong showing this page ({this.state.error.message}). Try another page, or reload.</p>
    ) : this.props.children;
  }
}

/** The header's status lines: the season, the last gameweek played and when it ended, and the next deadline. */
function Status({ site }: { site: Site }) {
  const { meta } = site;
  const last = Math.max(0, ...meta.played);
  // FPL doesn't say when a gameweek ended: take the last match's kick-off plus two hours, about the final whistle.
  const kickoffs = site.fixtures.filter((f) => f.gw === last && f.kickoff).map((f) => Date.parse(f.kickoff!));
  const ended = kickoffs.length ? new Date(Math.max(...kickoffs) + 2 * 3600 * 1000).toISOString() : null;
  const updated = new Date(meta.generated).toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short", year: "numeric" });
  return (
    <div className="status-row">
      <div className="status">
        <span><strong>{meta.season} Season</strong></span>
        <span>
          {last > 0 && <>Results up to Gameweek {last}{ended && <> ended: {when(ended)}</>} · </>}Data Last Updated {updated}
        </span>
        {meta.next_deadline && <span>Next Gameweek {meta.next_gw} deadline: <strong>{when(meta.next_deadline)}</strong></span>}
      </div>
      {meta.next_deadline && <Countdown to={meta.next_deadline} />}
    </div>
  );
}

/** Days, hours, minutes and seconds to `to`, ticking every second. */
function Countdown({ to }: { to: string }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  const left = Math.max(0, Math.floor((Date.parse(to) - now) / 1000));
  if (left === 0) return <div className="countdown"><span className="countdown-label">Deadline passed</span></div>;
  const parts: [number, string][] = [
    [Math.floor(left / 86400), "days"], [Math.floor(left / 3600) % 24, "hrs"],
    [Math.floor(left / 60) % 60, "min"], [left % 60, "sec"],
  ];
  return (
    <div className="countdown" role="timer" aria-label="Time to the next deadline">
      <span className="countdown-label">Deadline in</span>
      <div className="countdown-units">
        {parts.map(([n, unit]) => (
          <span key={unit} className="countdown-unit">
            <b>{String(n).padStart(2, "0")}</b>
            <small>{unit}</small>
          </span>
        ))}
      </div>
    </div>
  );
}
