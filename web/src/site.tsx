// The data every page shares (meta.json and players.json), and hooks for the rest.

import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { load, rows, type Fixture, type Meta, type MidweekMatch, type Player, type Team } from "./data";

export interface Site {
  /** Where this season's data files are: "" for the current season, "seasons/2025-26/" for an earlier one. */
  root: string;
  meta: Meta;
  players: Player[];
  player: Map<number, Player>;
  team: Map<number, Team>;
  teamByCode: Map<number, Team>;
  fixtures: Fixture[];
  midweek: MidweekMatch[];
}

export const SiteContext = createContext<Site | null>(null);

export function useSite(): Site {
  const site = useContext(SiteContext);
  if (!site) throw new Error("useSite outside SiteContext");
  return site;
}

export function buildSite(meta: Meta, players: Player[], root = ""): Site {
  return {
    root,
    meta,
    players,
    player: new Map(players.map((p) => [p.id, p])),
    team: new Map(meta.teams.map((t) => [t.id, t])),
    teamByCode: new Map(meta.teams.map((t) => [t.code, t])),
    fixtures: rows<Fixture>(meta.fixtures),
    midweek: rows<MidweekMatch>(meta.midweek),
  };
}

/** A data file, loaded on first use: undefined while loading, null if it's missing. */
export function useData<T>(path: string | null): T | null | undefined {
  const [value, setValue] = useState<{ path: string | null; data: T | null } | undefined>(undefined);
  useEffect(() => {
    let live = true;
    if (path === null) return;
    load<T>(path).then((data) => live && setValue({ path, data }));
    return () => {
      live = false;
    };
  }, [path]);
  return useMemo(() => (value && value.path === path ? value.data : undefined), [value, path]);
}

/** Loads meta.json and players.json. */
export function useSiteData(): Site | null | undefined {
  const meta = useData<Meta>("meta.json");
  const players = useData<Record<string, unknown[]>>("players.json");
  return useMemo(() => {
    if (meta === undefined || players === undefined) return undefined;
    if (!meta || !players) return null;
    return buildSite(meta, rows<Player>(players));
  }, [meta, players]);
}

/** The site for `season`: the current one as it is, an earlier one from its own copy in
 * seasons/<season>/ (its own clubs, players and fixtures: FPL renumbers ids every season).
 * undefined while loading, null if that season isn't there. */
export function useSeason(season: string | null): Site | null | undefined {
  const current = useSite();
  const past = season !== null && season !== current.meta.season ? `seasons/${season}/` : null;
  const meta = useData<Meta>(past && `${past}meta.json`);
  const players = useData<Record<string, unknown[]>>(past && `${past}players.json`);
  return useMemo(() => {
    if (!past) return current;
    if (meta === undefined || players === undefined) return undefined;
    if (!meta || !players) return null;
    return buildSite(meta, rows<Player>(players), past);
  }, [past, current, meta, players]);
}

/** "#gameweeks/6", or "#gameweeks/2025-26/6" for an earlier season; with a fixture, its match page. */
export function gameweekHref(site: Site, gw: number, fixture?: number): string {
  const season = site.meta.past ? `${site.meta.season}/` : "";
  return `#gameweeks/${season}${gw}${fixture !== undefined ? `/${fixture}` : ""}`;
}

/** The URL hash without the "#": "gameweeks/6/53". */
export function useHash(): string {
  const [hash, setHash] = useState(() => window.location.hash.slice(1));
  useEffect(() => {
    const onChange = () => setHash(window.location.hash.slice(1));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return hash;
}
