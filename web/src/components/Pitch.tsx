// The dashboard's pitch (app.py show_pitch), for the website: the XI by position on a striped
// pitch, then the bench in auto-sub order. Each card has the player's photo (the club shirt where
// the Premier League has none), the name on the club's colour, the fixture shaded by FPL's
// difficulty, the club's cup or European match that week if it has one, and a line of numbers.

import { useState, type ReactNode } from "react";
import { clubMidweek, competition, describe, matchDay } from "../midweek";
import { when } from "../format";
import { useSite } from "../site";

const PHOTO_URL = (code: number) => `https://resources.premierleague.com/premierleague25/photos/players/110x140/${code}.png`;
const SHIRT_URL = (teamCode: number, keeper: boolean) =>
  `https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_${teamCode}${keeper ? "_1" : ""}-66.png`;
const FDR_BG: Record<number, string> = { 1: "#cde2fb", 2: "#9ec5f4", 3: "#6da7ec", 4: "#256abf", 5: "#104281" };
const FDR_FG: Record<number, string> = { 1: "#0b0b0b", 2: "#0b0b0b", 3: "#0b0b0b", 4: "#ffffff", 5: "#ffffff" };

export interface PitchProps {
  gw: number;
  lineup: number[];
  bench: number[];
  captain?: number;
  vice?: number;
  /** The numbers under a player's name and fixture (e.g. "4.36 xP · £5.4"). Wrap the part after
   * the main number in `<span className="xp-extra">`: phones hide it. */
  line: (element: number) => ReactNode;
  /** Players to mark with a dashed outline (e.g. the hindsight-best XI). */
  marked?: Set<number>;
  /** Show FPL's injury flags (only meaningful for the next gameweek). */
  flags?: boolean;
}

export function Pitch({ gw, lineup, bench, captain, vice, line, marked, flags }: PitchProps) {
  const site = useSite();
  const card = (p: number, order?: string) => (
    <Card key={p} element={p} gw={gw} order={order} captain={p === captain} vice={p === vice}
          line={line(p)} marked={!!marked?.has(p)} flag={flags} />
  );
  const position = (p: number) => site.player.get(p)?.element_type;
  return (
    <div className="xp-pitch-wrap">
      <div className="xp-pitch">
        {[1, 2, 3, 4].map((pos) => (
          <div className="xp-row" key={pos}>{lineup.filter((p) => position(p) === pos).map((p) => card(p))}</div>
        ))}
      </div>
      <div className="xp-bench">
        <div className="xp-row">{bench.map((p, i) => card(p, i === 0 ? "GKP" : String(i)))}</div>
      </div>
    </div>
  );
}

function Card({ element, gw, order, captain, vice, line, marked, flag }: {
  element: number; gw: number; order?: string; captain: boolean; vice: boolean; line: ReactNode; marked: boolean; flag?: boolean;
}) {
  const site = useSite();
  const [photoMissing, setPhotoMissing] = useState(false);
  const p = site.player.get(element);
  if (!p) return null;
  const team = site.team.get(p.team);
  const [club, clubText] = (team && site.meta.club_colours[team.short]) ?? ["#d9dde3", "#111111"];
  const shirt = team ? SHIRT_URL(team.code, p.element_type === 1) : undefined;

  // The fixture(s) this gameweek, shaded by FPL's difficulty (a double takes the rounded mean).
  const fixtures = site.fixtures.filter((f) => f.gw === gw && (f.home === p.team || f.away === p.team));
  const label = fixtures.map((f) => f.home === p.team ? `${site.team.get(f.away)?.short} (H)` : `${site.team.get(f.home)?.short} (A)`).join(", ");
  const level = fixtures.length
    ? Math.round(fixtures.reduce((s, f) => s + (f.home === p.team ? f.home_fdr : f.away_fdr), 0) / fixtures.length) : null;
  const chance = p.chance_of_playing_next_round;
  const midweek = clubMidweek(site, p.team, gw);

  return (
    <div className={`xp-card${marked ? " xp-marked" : ""}`}
         title={`${p.first_name} ${p.second_name} · ${team?.name ?? ""} · £${p.now_cost.toFixed(1)}m`}>
      {order !== undefined && <div className="xp-order">{order}</div>}
      {captain ? <span className="xp-badge xp-cap">C</span> : vice ? <span className="xp-badge xp-vc">V</span> : null}
      {flag && chance !== null && chance < 100 && (
        <span className="xp-badge xp-flag" style={{ background: chance <= 25 ? "#d0021b" : "#f5a623" }} title={p.news}>{chance}</span>
      )}
      {photoMissing || !shirt ? (
        <div className="xp-face xp-shirt" style={{ borderColor: club, backgroundImage: shirt ? `url(${shirt})` : undefined }} />
      ) : (
        <div className="xp-face" style={{ borderColor: club }}>
          <img className="xp-photo" src={PHOTO_URL(p.code)} alt="" loading="lazy" onError={() => setPhotoMissing(true)} />
          <img className="xp-crest" src={shirt} alt="" loading="lazy" />
        </div>
      )}
      <div className="xp-name" style={{ background: club, color: clubText }}>{p.web_name}</div>
      <div className="xp-fix" style={level ? { background: FDR_BG[level], color: FDR_FG[level] } : undefined}>{label || "no fixture"}</div>
      {midweek.length > 0 && (
        <div className="xp-mid" title={midweek.map((m) => `${competition(m.tournament).name}, ${when(m.kickoff)}: ${describe(site, m)}`).join("\n")}>
          {midweek.map((m) => `${competition(m.tournament).short} ${matchDay(m.kickoff)}`.trim()).join(", ")} midweek
        </div>
      )}
      <div className="xp-pts">{line}</div>
    </div>
  );
}
