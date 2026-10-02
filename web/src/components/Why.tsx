// "Why this projection?": what goes into a player's forecast, grouped the way an FPL manager would
// weigh it up. The groups mirror the inputs this model reads (features.py): minutes and role, the
// underlying numbers, the fixture (club ratings), the betting market and the crowd. xP comes from
// the model as a whole, so these are its inputs, not a sum that adds up to it.

import { useMemo, type ReactNode } from "react";
import type { Forecast, Player } from "../data";
import { POSITIONS, STATUS, dec, money, pct, pts, risk, signed } from "../format";
import { ROTATION } from "../midweek";
import { fairResult, oddsUrl, type OddsSnapshot } from "../polymarket";
import { matchPlayer, ours } from "../ratings";
import { history, useAllGameweeks } from "../season";
import { useData, useSite } from "../site";
import { band } from "./Simulation";
import { Club, Loading, Opponent, Stats } from "./ui";

function Group({ title, hint, children }: { title: string; hint: string; children: ReactNode }) {
  return (
    <section className="why-group">
      <div className="why-head"><strong>{title}</strong><span>{hint}</span></div>
      {children}
    </section>
  );
}

function Meter({ value, label }: { value: number; label: string }) {
  return (
    <div className="meter" role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={value} aria-label={label}>
      <span style={{ width: `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%` }} />
    </div>
  );
}

const CONFIDENCE_TEXT = {
  High: "this model's three parts agree on him",
  Medium: "this model's three parts differ a little on him",
  Low: "this model's three parts disagree on him: forecasts like this have missed by more than usual",
};

/** "High confidence": how far this model's members agree on his next-gameweek xP (predict.py). */
export function ConfidenceBadge({ level, sd }: { level?: Forecast["confidence"]; sd?: number | null }) {
  if (!level) return null;
  return (
    <span className={`badge conf-${level.toLowerCase()}`}
          title={`${level} confidence: ${CONFIDENCE_TEXT[level]}${sd != null ? ` (their spread: ±${dec(sd)} points)` : ""}`}>
      {level} confidence
    </span>
  );
}

/** "Medium risk": how often the pick scores 2 or fewer (format.risk). */
export function RiskBadge({ pBlank }: { pBlank?: number | null }) {
  const level = risk(pBlank);
  if (!level) return null;
  return <span className="badge risk" title={`${pct(pBlank)} chance of 2 points or fewer`}>{level} risk</span>;
}

function Tile({ label, value, note }: { label: string; value: ReactNode; note?: ReactNode }) {
  return (
    <div className="why-tile">
      <div className="why-tile-label">{label}</div>
      <div className="why-tile-value">{value}</div>
      {note && <div className="why-tile-note">{note}</div>}
    </div>
  );
}

/** The next gameweek's four outcome bands as one bar: 0-2, 3-5, 6-9 and 10+ points. */
function OutcomeBands({ f, gw }: { f: Forecast; gw: number }) {
  const bands = [
    { label: "0–2 bust", p: f.p_blank, step: 25 },
    { label: "3–5 floor", p: f.p_3_5, step: 45 },
    { label: "6–9 middle", p: f.p_6_9, step: 70 },
    { label: "10+ haul", p: f.p_haul, step: 100 },
  ];
  if (bands.some((b) => b.p == null)) return null;
  const fill = (step: number) => `color-mix(in srgb, var(--s1) ${step}%, var(--chip))`;
  return (
    <div className="bands" role="group" aria-label={`GW${gw} outcome chances`}>
      <div className="why-tile-label" style={{ marginBottom: 4 }}>GW{gw} outcome chances</div>
      <div className="bands-bar" aria-hidden>
        {bands.map((b) => <span key={b.label} style={{ flexGrow: b.p!, background: fill(b.step) }} />)}
      </div>
      <div className="bands-labels">
        {bands.map((b) => (
          <div key={b.label}>
            <span className="key key-rect" style={{ background: fill(b.step) }} />
            <small>{b.label}</small>
            <b>{pct(b.p)}</b>
          </div>
        ))}
      </div>
    </div>
  );
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

/** The forecast for `player` in gameweek `gw` and what went into it. `forecast` is his row of the
 * saved forecast (next.json), when the page has it; otherwise the players file's next-GW xP. */
export function WhyProjection({ player, forecast, gw, gameweeks }: {
  player: Player; forecast?: Forecast | null; gw: number; gameweeks?: number[];
}) {
  const site = useSite();
  const odds = useData<OddsSnapshot>(oddsUrl());
  const all = useAllGameweeks(site.meta.played);
  const games = useMemo(() => history(all, player.id), [all, player.id]);
  const xp = forecast ? forecast[`xp_${gw}`] : player.forecast;
  const last5 = games.slice(-5);
  const per90 = (v: number) => (player.minutes > 0 ? (v * 90) / player.minutes : null);
  const code = (id: number) => site.team.get(id)!.code;

  // His club's matches in the gameweek: this model's ratings, and the market's prices where it has them.
  const matches = site.fixtures.filter((f) => f.gw === gw && (f.home === player.team || f.away === player.team)).map((f) => {
    const home = f.home === player.team;
    const o = ours(site, code(f.home), code(f.away));
    const gf = home ? o.ours_home : o.ours_away, ga = home ? o.ours_away : o.ours_home;
    const market = odds?.matches.find((m) => m.home_code === code(f.home) && m.away_code === code(f.away));
    const fair = market ? fairResult(market.prices) : null;
    const scorer = market ? odds?.scorers.find((s) => s.slug === market.slug && matchPlayer(site, s)?.id === player.id) : undefined;
    return {
      id: f.id, opponent: home ? f.away : f.home, home,
      win: home ? o.ours_home_win : o.ours_away_win, gf, ga, cs: ga === null ? null : Math.exp(-ga),
      market: market && {
        win: fair?.[home ? "home_win" : "away_win"] ?? null,
        gf: home ? market.lam_home : market.lam_away, ga: home ? market.lam_away : market.lam_home,
        scorer: scorer && scorer.volume > 0 ? scorer.p : null,
      },
    };
  });
  const status = STATUS[player.status];
  const total = forecast && gameweeks && gameweeks.length > 1 ? forecast.xp_total : null;
  const defender = player.element_type <= 2;
  const n = gameweeks?.length ?? 1;
  const first3 = forecast && gameweeks && n > 3 ? gameweeks.slice(0, 3).reduce((a, g) => a + (forecast[`xp_${g}`] ?? 0), 0) : null;
  const level = risk(forecast?.p_blank);

  return (
    <div className="why">
      <div className="why-badges">
        <ConfidenceBadge level={forecast?.confidence} sd={forecast?.xp_sd} />
        <RiskBadge pBlank={forecast?.p_blank} />
        <span className="muted">{site.team.get(player.team)?.name} · {POSITIONS[player.element_type]} · {money(player.now_cost)}
          {status ? <> · <span className="bad">{status}</span></> : " · Available"}</span>
      </div>

      <div className="why-tiles why-tiles-xp">
        <Tile label="1 GW" value={pts(xp)} note={`xP in GW${gw}`} />
        {first3 !== null && <Tile label="3 GW" value={pts(first3)} note={`GW${gameweeks![0]}–${gameweeks![2]}`} />}
        {total !== null && <Tile label={`${n} GW`} value={pts(total)} note={`GW${gameweeks![0]}–${gameweeks![n - 1]}`} />}
        {level && <Tile label="Risk" value={level} note={`${pct(forecast?.p_blank)} chance of 2 or fewer`} />}
      </div>

      {forecast?.pts_p90 != null && (
        <div className="why-tiles">
          <Tile label="1 GW range" value={band(forecast.pts_p10, forecast.pts_p90)} note="the middle 80% of simulated weeks" />
          {forecast.total_p90 != null && n > 1 && <>
            <Tile label={`${n} GW floor`} value={dec(forecast.total_p10, 0)} note={`1 in 10 simulated runs of GW${gw}–${gameweeks![n - 1]} end below this`} />
            <Tile label={`${n} GW median`} value={dec(forecast.total_p50, 0)} note="the middle outcome" />
            <Tile label={`${n} GW ceiling`} value={dec(forecast.total_p90, 0)} note="1 in 10 runs end above this" />
          </>}
        </div>
      )}

      {forecast?.xmins != null && (
        <div className="why-tiles">
          <Tile label="Expected minutes" value={dec(forecast.xmins, 0)} note={`in GW${gw}, flags and rotation included`} />
          {forecast.p_play != null && <Tile label="Plays" value={pct(forecast.p_play)} note="any minutes" />}
          {forecast.p_full != null && <Tile label="60+ minutes" value={pct(forecast.p_full)} note="roughly: starts" />}
          {forecast.p_full != null && forecast.p_play != null &&
            <Tile label="Cameo" value={pct(Math.max(0, forecast.p_play - forecast.p_full))} note="on for under an hour" />}
        </div>
      )}

      {forecast && <OutcomeBands f={forecast} gw={gw} />}
      {forecast?.confidence && (
        <p className="note" style={{ marginTop: 0 }}>
          <strong>{forecast.confidence} confidence:</strong> {CONFIDENCE_TEXT[forecast.confidence]}
          {forecast.xp_sd != null && <> (their forecasts spread by ±{dec(forecast.xp_sd)} points)</>}. Tested on six past seasons
          (<a href="#about/report/accuracy">how</a>). Risk is about the range of outcomes; confidence is about the forecast itself.
        </p>
      )}

      <Group title="Minutes and role" hint="his place in the team">
        {forecast?.p_play != null && <Meter value={forecast.p_play} label="Chance he plays" />}
        <Stats items={[
          ...(forecast?.p_play != null && forecast.xmins == null ? [{ label: "Chance he plays", value: pct(forecast.p_play) }] : []),
          { label: "Starts this season", value: `${player.starts} of ${site.meta.played.length}` },
          ...(last5.length ? [{ label: `Minutes, last ${last5.length}`, value: dec(mean(last5.map((g) => g.minutes)), 0) + " a game" }] : []),
          { label: "FPL flag", value: status ? <span className="bad">{status}{player.chance_of_playing_next_round != null && ` ${player.chance_of_playing_next_round}%`}</span> : "None" },
          ...(forecast?.rotation ? [{ label: "Midweek factor", value: <span className={(forecast.rotation_factor ?? 1) >= 1 ? "good" : "bad"}>×{dec(forecast.rotation_factor)}</span>,
                                       title: ROTATION[forecast.rotation] ?? forecast.rotation }] : []),
        ]} />
        {player.news && <p>{player.news}</p>}
      </Group>

      <Group title="Underlying numbers" hint="what his play is worth">
        <Stats items={[
          { label: "xG per 90", value: dec(per90(player.expected_goals)) },
          { label: "xA per 90", value: dec(per90(player.expected_assists)) },
          { label: "Goals + assists", value: `${player.goals_scored} + ${player.assists}` },
          ...(defender || player.element_type === 3 ? [{ label: "Def. contribution per 90", value: dec(per90(player.defensive_contribution), 1) }] : []),
          ...(last5.length ? [
            { label: `Points, last ${last5.length}`, value: `${dec(mean(last5.map((g) => g.points)), 1)} a game` },
            { label: `xP, last ${last5.length}`, value: (() => {
              const xs = last5.map((g) => g.xp).filter((x): x is number => x !== null);
              return xs.length ? `${dec(mean(xs), 1)} a game` : "–";
            })() },
          ] : []),
        ]} />
        {all === undefined && <Loading />}
      </Group>

      <Group title="Fixture" hint="this model's club ratings">
        {!matches.length && <p>No match in GW{gw} (a blank gameweek).</p>}
        {matches.map((m) => (
          <div key={m.id}>
            <p>Against <Opponent id={m.opponent} home={m.home} /></p>
            <Stats items={[
              { label: "Win chance", value: pct(m.win) },
              { label: "Goals for – against", value: m.gf === null ? "–" : `${dec(m.gf, 1)} – ${dec(m.ga, 1)}` },
              { label: "Clean-sheet chance", value: pct(m.cs), title: "No goals against, from the expected goals against" },
            ]} />
          </div>
        ))}
      </Group>

      <Group title="Betting market" hint="Polymarket, refreshed through the week">
        {odds === undefined ? <Loading /> : !matches.some((m) => m.market) ? (
          <p>No market for this match yet: match markets usually open a few days before kick-off. Without one, this model uses its own club ratings.</p>
        ) : matches.filter((m) => m.market).map((m) => (
          <Stats key={m.id} items={[
            { label: "Win chance", value: pct(m.market!.win) },
            { label: "Goals for – against", value: `${dec(m.market!.gf, 1)} – ${dec(m.market!.ga, 1)}` },
            { label: "To score", value: m.market!.scorer != null
                ? <span className={m.market!.scorer <= site.meta.out_threshold ? "bad" : undefined}>{pct(m.market!.scorer, 1)}</span>
                : "no market", title: `At ${pct(site.meta.out_threshold)} or less, this model treats him as ruled out` },
          ]} />
        ))}
      </Group>

      <Group title="Crowd and price" hint="what other managers are doing">
        <Stats items={[
          { label: "Owned by", value: `${dec(player.selected_by_percent, 1)}%` },
          { label: "FPL form", value: dec(player.form, 1), title: "FPL's form: points per match over the last 30 days" },
          { label: "Price", value: money(player.now_cost) },
          { label: "Since GW1", value: player.cost_change_start ? `${signed(player.cost_change_start / 10, 1)}m` : "–" },
        ]} />
      </Group>

      <p className="note">
        <Club id={player.team} /> {POSITIONS[player.element_type]}. These are the things this model weighs up; the xP comes from
        this model as a whole, so they don't add up to it. Managers buying and selling before the deadline are its strongest
        single input (<a href="#about/report">technical report</a>).
      </p>
    </div>
  );
}
