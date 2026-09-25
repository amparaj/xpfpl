// Per-gameweek player totals (a double gameweek's two matches summed) and loading every played
// gameweek at once.

import { useEffect, useState } from "react";
import { gwFile, load, rows, type Gameweek, type GwRow } from "./data";

export interface PlayerGw {
  element: number; gw: number; matches: number; opponents: { team: number; home: boolean }[];
  minutes: number; points: number; goals: number; assists: number; clean_sheets: number; bonus: number;
  bps: number; saves: number; yellow: number; red: number; xg: number | null; xa: number | null;
  dc: number | null; xp: number | null; price: number; selected: number; transfers: number;
  pre_chance: number | null; pre_news: string | null;
}

const add = (a: number | null, b: number | null | undefined) =>
  b === null || b === undefined ? a : (a ?? 0) + b;

const memo = new WeakMap<Gameweek, Map<number, PlayerGw>>();

export function totals(g: Gameweek): Map<number, PlayerGw> {
  const cached = memo.get(g);
  if (cached) return cached;
  const out = new Map<number, PlayerGw>();
  memo.set(g, out);
  for (const r of rows<GwRow>(g.players)) {
    let t = out.get(r.element);
    if (!t) {
      t = { element: r.element, gw: g.gw, matches: 0, opponents: [], minutes: 0, points: 0, goals: 0, assists: 0,
            clean_sheets: 0, bonus: 0, bps: 0, saves: 0, yellow: 0, red: 0, xg: null, xa: null, dc: null, xp: null,
            price: r.value, selected: r.selected, transfers: r.transfers_balance,
            pre_chance: r.pre_chance ?? null, pre_news: r.pre_news ?? null };
      out.set(r.element, t);
    }
    t.matches += 1;
    t.opponents.push({ team: r.opponent_team, home: r.was_home });
    t.minutes += r.minutes;
    t.points += r.total_points;
    t.goals += r.goals_scored;
    t.assists += r.assists;
    t.clean_sheets += r.clean_sheets;
    t.bonus += r.bonus;
    t.bps += r.bps;
    t.saves += r.saves;
    t.yellow += r.yellow_cards;
    t.red += r.red_cards;
    t.xg = add(t.xg, r.expected_goals);
    t.xa = add(t.xa, r.expected_assists);
    t.dc = add(t.dc, r.defensive_contribution);
    t.xp = add(t.xp, r.xp);
  }
  return out;
}

/** Every played gameweek's file, loaded together (undefined until all have arrived). */
export function useAllGameweeks(played: number[]): Map<number, Gameweek> | undefined {
  const [data, setData] = useState<Map<number, Gameweek>>();
  const key = played.join(",");
  useEffect(() => {
    let live = true;
    Promise.all(played.map((gw) => load<Gameweek>(gwFile(gw)))).then((files) => {
      if (live) setData(new Map(files.filter((f): f is Gameweek => !!f).map((f) => [f.gw, f])));
    });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return data;
}

/** A player's gameweek-by-gameweek totals across the season. */
export function history(all: Map<number, Gameweek> | undefined, element: number): PlayerGw[] {
  if (!all) return [];
  return [...all.values()].sort((a, b) => a.gw - b.gw).flatMap((g) => {
    const t = totals(g).get(element);
    return t ? [t] : [];
  });
}
