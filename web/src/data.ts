// Loading the JSON that `xpfpl export` writes to public/data/, and the shapes it has.

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
/** One gameweek of the Model's Team: the decision saved before the deadline and, once played,
 * what it scored. `source`: "live" (decided before the deadline), "replay" (filled in by the
 * backtest for weeks before the live record began) or "carried" (no decision saved: last week's team). */
export interface ModelWeek {
  gw: number; source: "live" | "replay" | "carried"; model: string; made_at: string; chip: string | null;
  free_transfers: number; bank: number; hits: number;
  transfers: { out: number; in: number; sold: number; bought: number }[];
  squad: number[]; lineup: number[]; bench: number[]; captain: number; vice: number; xp: number | null;
  points?: number; gross?: number; captain_played?: number | null; autosubs?: [number, number][];
  best?: number; best_xi?: number[]; best_captain?: number;
}
export interface ModelTeam { model: string; gameweeks: ModelWeek[]; next: ModelWeek | null }
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
export interface Accuracy { validation: any; comparison: any; tuning: any; scorecard: any; rotation?: RotationFit | null }

const cache = new Map<string, Promise<any>>();

/** A file from public/data/ (or a full URL), fetched once. Resolves to null if it doesn't exist. */
export function load<T>(path: string): Promise<T | null> {
  if (!cache.has(path)) {
    cache.set(path, fetch(/^https?:/.test(path) ? path : `./data/${path}`).then((r) => (r.ok ? r.json() : null)).catch(() => null));
  }
  return cache.get(path)!;
}

export const gwFile = (gw: number) => `gws/gw${String(gw).padStart(2, "0")}.json`;
