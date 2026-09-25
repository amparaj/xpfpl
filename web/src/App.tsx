import { Component, useEffect, useState, type ReactNode } from "react";
import { Loading } from "./components/ui";
import { day, when } from "./format";
import About from "./pages/About";
import Accuracy from "./pages/Accuracy";
import DataPage from "./pages/Data";
import Gameweeks from "./pages/Gameweeks";
import MarketsPage from "./pages/Markets";
import MyTeam from "./pages/MyTeam";
import Players from "./pages/Players";
import { SiteContext, useSiteData } from "./site";

// About first (where the site opens), then look back at the week, research, one team's season,
// how far to trust it, and the raw data last.
const PAGES = [
  { id: "about", label: "About", component: About },
  { id: "gameweeks", label: "Gameweeks", component: Gameweeks },
  { id: "players", label: "Players", component: Players },
  { id: "markets", label: "Markets", component: MarketsPage },
  { id: "team", label: "My Team", component: MyTeam },
  { id: "accuracy", label: "Model Accuracy", component: Accuracy },
  { id: "data", label: "Data", component: DataPage },
] as const;

/** The page is the part of the URL hash before any "/": #players/123 -> players. */
function useHash(): string {
  const [hash, setHash] = useState(() => window.location.hash.slice(1));
  useEffect(() => {
    const onChange = () => setHash(window.location.hash.slice(1));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return hash;
}

export default function App() {
  const site = useSiteData();
  const hash = useHash();
  const page = PAGES.find((p) => p.id === hash.split("/")[0]) ?? PAGES[0];
  const Page = page.component;

  useEffect(() => {
    document.title = `${page.label} · xP-FPL`;
  }, [page]);

  return (
    <>
      <header className="top">
        <div className="top-inner">
          <div className="brand">
            <h1>xP-FPL</h1>
            {site && (
              <span className="muted">
                {site.meta.season} · data to GW{Math.max(0, ...site.meta.played)} · updated {day(site.meta.generated)}
                {site.meta.next_deadline && <> · GW{site.meta.next_gw} deadline {when(site.meta.next_deadline)}</>}
              </span>
            )}
          </div>
          <nav aria-label="Pages">
            {PAGES.map((p) => (
              <a key={p.id} href={`#${p.id}`} className={p === page ? "on" : undefined}
                 aria-current={p === page ? "page" : undefined}>
                {p.label}
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
        Expected points (xP) from a PyTorch model trained on every Premier League season since 2016-17. Betting odds
        from Polymarket. Not affiliated with the Premier League or Fantasy Premier League.
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
