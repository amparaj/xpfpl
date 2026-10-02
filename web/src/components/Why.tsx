// "Why this projection?": what goes into a player's forecast, grouped the way an FPL manager would
// weigh it up. The groups mirror the inputs this model reads (features.py): minutes and role, the
// underlying numbers, the fixture (club ratings), the betting market and the crowd. xP comes from
// the model as a whole, so these are its inputs, not a sum that adds up to it.

import { useMemo, type ReactNode } from "react";
import type { Forecast, Player } from "../data";
import { POSITIONS, STATUS, dec, money, pct, pts, signed } from "../format";
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

  return (
    <div className="why">
      <div>
        <div className="why-xp">
          <b>{pts(xp)}</b>
          <span>xP in GW{gw}</span>
          {total !== null && <span>· {pts(total)} over GW{gameweeks![0]}–{gameweeks![gameweeks!.length - 1]}</span>}
        </div>
        {forecast?.pts_p90 != null && (
          <p className="note" style={{ marginTop: 0 }}>
            Simulated thousands of times: {band(forecast.pts_p10, forecast.pts_p90)} points in 4 weeks out of 5,{" "}
            {pct(forecast.p_haul)} chance of 10+, {pct(forecast.p_blank)} of 2 or fewer.
          </p>
        )}
      </div>

      <Group title="Minutes" hint="will he play, and for how long?">
        {forecast?.p_play != null && <Meter value={forecast.p_play} label="Chance he plays" />}
        <Stats items={[
          ...(forecast?.p_play != null ? [{ label: "Chance he plays", value: pct(forecast.p_play) }] : []),
          ...(forecast?.xmins != null ? [{ label: "Expected minutes", value: dec(forecast.xmins, 0) }] : []),
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
