// Loading the JSON that `xpfpl export` writes to public/data/, and the shapes it has.

import type { Sealed } from "./seal";

/** A column-wise table as exported: {"col": [values...]}. */
export type Columns = Record<string, unknown[]>;
export type Row = Record<string, any>;

/** Column-wise -> one object per row. */
export function rows<T = Row>(cols: Columns | undefined | null): T[] {
  if (!cols) return [];
  const keys = Object.keys(cols);
  const n = keys.length ? cols[keys[0]].length : 0;
  const out: T[] = [];
  for (let i = 0; i < n; i++) {
    const r: Row = {};
    for (const k of keys) r[k] = cols[k][i];
    out.push(r as T);
  }
  return out;
}

export interface Team { id: number; code: number; name: string; short: string }
export interface Event {
  id: number; deadline: string; finished: boolean; checked: boolean;
  average: number | null; highest: number | null; most_captained: number | null; top_element: number | null;
}
export interface Fixture {
  id: number; gw: number | null; kickoff: string | null; home: number; away: number;
  home_score: number | null; away_score: number | null; home_fdr: number; away_fdr: number;
}
export interface Meta {
  generated: string; season: string; model: string; model_description: string;
  played: number[]; next_gw: number | null; next_deadline: string | null;
  teams: Team[]; events: Event[]; chips: { name: string; start: number; stop: number }[];
  fixtures: Columns; club_colours: Record<string, [string, string]>;
  polymarket_teams: Record<string, number>; out_threshold: number;
  ratings_next: { gw: number; mu: number; home: number; prior: [number, number]; clubs: Record<string, [number, number]> } | Record<string, never>;
  repo: string | null;
  /** This season's cup and European matches (data/cups.py), filed under the gameweek they come before. */
  midweek?: Columns;
  /** The seasons that can be looked back at, oldest first, this one last (current meta.json only). */
  seasons?: string[];
  /** Set in an earlier season's meta.json (seasons/<season>/). */
  past?: boolean;
}
/** A cup or European match. `home_code`/`away_code` are FPL club codes for Premier League clubs. */
export interface MidweekMatch {
  match_id: string; gw: number; kickoff: string | null; tournament: string;
  home_code: number | null; away_code: number | null; home_name: string; away_name: string;
  home_score: number | null; away_score: number | null; finished: boolean;
}
export interface Player {
  id: number; code: number; web_name: string; first_name: string; second_name: string; team: number;
  element_type: number; now_cost: number; selected_by_percent: number; status: string; news: string;
  chance_of_playing_next_round: number | null; total_points: number; minutes: number; goals_scored: number;
  assists: number; clean_sheets: number; bonus: number; expected_goals: number; expected_assists: number;
  defensive_contribution: number; starts: number; form: number; points_per_game: number;
  cost_change_start: number; forecast: number | null;
}
export interface GwRow {
  element: number; fixture: number; opponent_team: number; was_home: boolean; minutes: number;
  total_points: number; goals_scored: number; assists: number; clean_sheets: number; goals_conceded: number;
  own_goals: number; penalties_saved: number; penalties_missed: number; yellow_cards: number; red_cards: number;
  saves: number; bonus: number; bps: number; expected_goals: number | null; expected_assists: number | null;
  expected_goals_conceded: number | null; defensive_contribution: number | null; starts: number | null;
  value: number; selected: number; transfers_balance: number; xp: number | null;
  pre_chance?: number | null; pre_news?: string | null; midweek_minutes?: number | null;
}
export interface Gameweek {
  gw: number; xp_source: string; players: Columns;
  fixtures: { id: number; kickoff: string; home: number; away: number; home_score: number | null; away_score: number | null }[];
}
/** Simulated scores (simulate.spread): mean, percentiles and a histogram in `width`-point buckets. */
export interface Spread {
  mean: number; p5: number; p10: number; p25: number; p50: number; p75: number; p90: number; p95: number;
  histogram?: { start: number; width: number; shares: number[] };
}
/** A captain option's simulated points (his own, before the armband) and how often he's the best pick. */
export interface CaptainOdds {
  element: number; mean: number; pts_p10: number; pts_p50: number; pts_p90: number;
  p_haul: number; p_blank: number; p_best: number; armband_mean: number;
}
/** What Triple Captain / Bench Boost would add this week in simulation, and the chance it clears the threshold. */
export interface ChipOdds extends Omit<Spread, "histogram"> { threshold: number; p_clear: number; p_beats_later: number | null }
/** The Monte Carlo saved with a Model's Team decision (modelteam.simulation). */
export interface Simulation { sims: number; points: Spread; captains: CaptainOdds[]; chips: Record<string, ChipOdds> }

/** One gameweek of the Model's Team: the decision saved before the deadline and, once played,
 * what it scored. `source`: "live" (decided before the deadline), "replay" (filled in by the
 * backtest for weeks before the live record began) or "carried" (no decision saved: last week's team). */
export interface ModelWeek {
  gw: number; source: "live" | "replay" | "carried"; model: string; made_at: string; chip: string | null;
  free_transfers: number; bank: number; hits: number;
  transfers: { out: number; in: number; sold: number; bought: number }[];
  squad: number[]; lineup: number[]; bench: number[]; captain: number; vice: number; xp: number | null;
  /** The team's forecast counted the way the week is scored (captain doubled, bench with Bench Boost,
   * before hits): from the decision's own xP, or summed from the week's player xP for a carried-over week. */
  forecast?: number | null; forecast_source?: "decision" | "gameweek xP" | null; bench_xp?: number;
  points?: number; gross?: number; captain_played?: number | null; autosubs?: [number, number][];
  best?: number; best_xi?: number[]; best_captain?: number;
  /** The Monte Carlo made with the decision: the team's simulated score, captain odds, chip odds. */
  simulation?: Simulation | null;
}
export interface ModelTeam { model: string; gameweeks: ModelWeek[]; next: ModelWeek | null }
/** One gameweek of My Team's outlook as anyone sees it: the forecast (xP counted the way the week is
 * scored) and the simulated score before transfer hits. */
export interface MyWeekNumbers { gw: number; xp: number; points: Spread | null; target?: number; p_target?: number }
/** ...and with the team (myteam.outlook): the first week as picked, later ones the same 15 with
 * their best XI and captain. */
export interface MyWeek extends MyWeekNumbers {
  lineup: number[]; bench: number[]; captain: number; vice: number; chip: string | null; bench_xp: number;
}
/** The part of My Team that's hidden before the deadline. `source`: "saved" (picked in the dashboard),
 * "carried" (nothing saved: last week's team) or "locked" (the deadline has passed: the team in FPL). */
export interface PlanSettings {
  horizon: number; model: string; chip: string | null; free_transfers: number; free_transfers_estimated: boolean;
  bank: number; max_hits: number; plan_transfers: boolean; value_prices: boolean; must_have: number[]; banned: number[];
}
export interface MyTeamPrivate {
  source: "saved" | "carried" | "locked"; saved_at: string | null; chip: string | null;
  transfers: { out: number; in: number }[]; hits: number; bank: number | null;
  /** The Plan Ahead settings the team was last saved with (myteam.save_plan); null for older saves. */
  settings?: PlanSettings | null;
  weeks: MyWeek[]; captains: CaptainOdds[];
  /** The squad: xP per gameweek (`xp_<gw>`, `xp_total`) and this gameweek's simulated range. */
  players: Columns;
}
/** My Team (export._my_team): the team for the gameweek in progress, until it's finished, or the next
 * one. The numbers are public. The team itself is `private` once the deadline has passed; before it,
 * `sealed` (encrypted: seal.ts opens it with the secret word) or absent. */
export interface MyTeam {
  gw: number; deadline: string; locked: boolean; model: string; team_name: string; gameweeks: number[];
  sims: number | null; weeks: MyWeekNumbers[];
  /** Every week added up, after any transfer hits. */
  total: { xp: number; points: Spread | null };
  /** The team against keeping last week's squad (its best XI and captain each week), in the same simulated weeks. */
  against: { xp: number; p_better?: number; p_tie?: number; mean?: number; p10?: number; p50?: number; p90?: number } | null;
  private: MyTeamPrivate | null;
  sealed: Sealed | null;
}
/** One player's row of the saved forecast: xP for each gameweek of the horizon (`xp_<gw>`), and
 * for the next gameweek his minutes, midweek factor and simulated range where the model gives them. */
export interface Forecast {
  element: number; xp_total: number; p_play?: number | null; xmins?: number | null;
  rotation?: string | null; rotation_factor?: number | null;
  /** The next gameweek's Monte Carlo: 10th/50th/90th percentile of his simulated points, chance of 10+ and of 2 or fewer. */
  pts_p10?: number | null; pts_p50?: number | null; pts_p90?: number | null; p_haul?: number | null; p_blank?: number | null;
  /** The other two outcome bands of the next gameweek (with p_blank and p_haul they add to 1). */
  p_3_5?: number | null; p_6_9?: number | null;
  /** The whole horizon's simulated total: floor (10th percentile), middle and ceiling (90th). */
  total_p10?: number | null; total_p50?: number | null; total_p90?: number | null;
  /** His chance of 60+ minutes in the next gameweek (p_play less this is a cameo). */
  p_full?: number | null;
  /** How far the ensemble's members disagree on his next-gameweek xP, and the confidence that gives. */
  xp_sd?: number | null; confidence?: "High" | "Medium" | "Low" | null;
  [xp: `xp_${number}`]: number;
}
/** The forecast saved for the next gameweek: xP per player for each gameweek of its horizon. */
export interface NextGw { gw: number; deadline: string; model: string; gameweeks: number[]; players: Columns }
export interface Markets {
  matches: Columns; accuracy: Columns; scorers?: Columns;
  outrights?: Columns & { snapshot: string };
}
/** The midweek rotation factors on next week's xP, per group (data/cups.py `fit`). */
export interface RotationFit {
  seasons: string[];
  groups: Record<string, { rows: number; points: number; xp: number; ratio: number | null; factor: number }>;
}
export interface Accuracy {
  validation: any; comparison: any; tuning: any; scorecard: any; rotation?: RotationFit | null;
  /** `xpfpl robustness`: six seasons, each forecast and replayed by a model trained only on earlier ones. */
  robustness?: any;
}

const cache = new Map<string, Promise<any>>();

/** A file from public/data/ (or a full URL), fetched once. Resolves to null if it doesn't exist. */
export function load<T>(path: string): Promise<T | null> {
  if (!cache.has(path)) {
    cache.set(path, fetch(/^https?:/.test(path) ? path : `./data/${path}`).then((r) => (r.ok ? r.json() : null)).catch(() => null));
  }
  return cache.get(path)!;
}

export const gwFile = (gw: number) => `gws/gw${String(gw).padStart(2, "0")}.json`;
