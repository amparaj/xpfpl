// The gameweek in progress against its forecast (src/xpfpl/live.py): live.json from the `live`
// branch (refreshed every ten minutes during matches by .github/workflows/live-snapshot.yml) or
// the copy published with the site, whichever is newer.

import { useMemo } from "react";
import type { Columns, Spread } from "./data";
import { useData } from "./site";

/** FPL's events in a match, per side ("h", "a") as [element, value] pairs. */
export type LiveEvents = Partial<Record<"goals_scored" | "assists" | "own_goals" | "penalties_saved" | "penalties_missed" |
  "yellow_cards" | "red_cards" | "saves" | "bonus", { h: [number, number][]; a: [number, number][] }>> & {
  /** The match's BPS leaders: [element, BPS, side]. */
  bps?: [number, number, "h" | "a"][];
  /** The bonus was worked out from the BPS: FPL hasn't awarded it yet. */
  bonus_provisional?: boolean;
};

export interface LiveMatch {
  id: number; home: number; away: number; kickoff: string | null;
  home_score: number | null; away_score: number | null; minutes: number | null; state: "done" | "live" | "to play";
  events?: LiveEvents;
}

/** Players' average over a group: points minus xP split into minutes, finishing and the rest. */
export interface LiveSummary {
  n: number; xp?: number; points?: number; mae?: number; bias?: number;
  minutes?: number; finishing?: number; other?: number;
  inside?: number; above?: number; below?: number;
  hauls_expected?: number; hauls?: number; blanks_expected?: number; blanks?: number;
}

export interface LivePick {
  element: number; role: "C" | "V" | "XI" | "Bench"; counted: boolean; xp: number; points: number;
  minutes: number; state: string; multiplier: number;
}

export interface LiveTeam {
  points: number; forecast: number | null; difference: number | null; still_to_play: number; still_xp: number;
  projected: number; settled: boolean; percentile: number | null; spread: Partial<Pick<Spread, "p10" | "p50" | "p90">>;
  captain: number; armband: number | null; subs: { out: number; in: number }[]; chip: string | null;
  made_at: string | null; picks: LivePick[];
}

export interface Live {
  generated: string; gw: number; deadline: string | null; model: string | null; forecast_made: string | null;
  matches: LiveMatch[]; done: number; total: number;
  week: LiveSummary & { positions?: Record<string, LiveSummary> };
  teams: { model?: LiveTeam; mine?: LiveTeam };
  lessons: Record<"stale" | "backed" | "lucky" | "unlucky" | "minutes", number[]>;
  stale: number;
  players: Columns;
}

export interface LivePlayer {
  element: number; team: number; xp: number; bps?: number; saves?: number; defensive_contribution?: number; points: number; minutes: number; state: string; p_play: number | null;
  pts_p10: number | null; pts_p90: number | null; p_haul: number | null; p_blank: number | null;
  goals_scored: number; assists: number; expected_goals: number; expected_assists: number; bonus: number;
  error: number; minutes_part: number; finishing: number; other: number; band: "inside" | "above" | "below" | null;
  news_changed: boolean; verdict: string | null; reasons: string[];
}

export function liveUrl(): string | null {
  const { hostname, pathname } = window.location;
  if (!hostname.endsWith(".github.io")) return null;
  const owner = hostname.slice(0, -".github.io".length), repo = pathname.split("/")[1];
  return `https://raw.githubusercontent.com/${owner}/${repo}/live/live.json`;
}

/** The newer of the branch's live.json and the site's own: undefined while loading, null if neither. */
export function useLive(): Live | null | undefined {
  const remote = useData<Live>(liveUrl());
  const local = useData<Live>("live.json");
  return useMemo(() => {
    const url = liveUrl();
    if (local === undefined || (url && remote === undefined)) return undefined;
    const all = [remote, local].filter((l): l is Live => !!l);
    if (!all.length) return null;
    return all.sort((a, b) => b.gw - a.gw || Date.parse(b.generated) - Date.parse(a.generated))[0];
  }, [remote, local]);
}
