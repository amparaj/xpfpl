import type { Accuracy } from "../data";
import { dec, int } from "../format";
import { ROTATION } from "../midweek";
import { useData, useSite } from "../site";

// The landing page: what the project is and how it works, in plain language. Every number comes
// from the exported reports, so it stays current each time the site is published.

const STEPS = [
  { title: "Collect", text: "Every player's stats in every Premier League match since 2016, plus betting odds." },
  { title: "Forecast", text: "A model predicts how many points each player will score in the next few gameweeks." },
  { title: "Pick", text: "An optimiser finds the best legal team, transfers, captain and chips for those forecasts." },
  { title: "Check", text: "After every gameweek, the forecasts are scored against what really happened." },
];

export default function About() {
  const site = useSite();
  const report = useData<Accuracy>("accuracy.json");
  const comparison = report?.comparison;
  const models: { model: string; mae: number }[] = comparison?.models ?? [];
  const mae = (name: string) => models.find((m) => m.model === name)?.mae;
  const ours = mae(site.meta.model);
  const baseline = mae("baseline");
  const fpl = mae("FPL xP");
  const ceiling = comparison?.ceiling?.mae_median as number | undefined;
  // Season replays (`xpfpl robustness`): each season played by a model trained only on the seasons before it.
  const replays = report?.robustness?.backtest;
  const seasonMean = (variant: string): number | undefined => {
    const bySeason = replays?.points?.[variant] as Record<string, number> | undefined;
    const values = bySeason ? Object.values(bySeason) : [];
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : undefined;
  };
  const replayModel = seasonMean("ensemble");
  const replayBaseline = seasonMean("baseline");
  const replaySeasons = report?.robustness?.seasons as string[] | undefined;
  const noise = replays?.noise_sd as number | undefined;
  const trainedOn = report?.validation?.trained_on as string | undefined;
  const rotation = report?.rotation;
  const groups = rotation ? Object.entries(rotation.groups).filter(([, g]) => g.rows > 0) : [];

  return (
    <article className="about">
      <h2>About</h2>
      <p className="lede">
        xP-FPL (Expected Points for FPL) is a personal project that predicts how many points every Fantasy Premier
        League player is likely to score in the coming gameweeks. A machine-learning model, trained on every Premier
        League season since 2016-17, makes the forecasts; an optimiser then turns them into a team each week: the
        starting eleven, captain, transfers and when to play a chip.
      </p>
      <p>
        This site is the project's public record for the {site.meta.season} season: what the model expected before each
        deadline, what actually happened, how accurate it has been, and how its own team is doing.
      </p>
      <p>
        "xP" means <strong>expected points</strong>: the average score a player would get if the same match could be
        played many times over. A striker on 6 xP won't score exactly 6. He might blank or get 15, but over a season
        those forecasts should add up.
      </p>

      <div className="steps">
        {STEPS.map((s, i) => (
          <div className="step" key={s.title}>
            <div className="step-number">{i + 1}</div>
            <div className="step-title">{s.title}</div>
            <div className="step-text">{s.text}</div>
          </div>
        ))}
      </div>

      <h3>1. The data</h3>
      <p>
        The model learns from every player in every Premier League match since the 2016-17 season: about a quarter of a
        million player-matches, with minutes, goals, assists, clean sheets, bonus points and, for recent seasons,
        expected goals (xG). It also uses odds from Polymarket, a betting market, which often reacts to team news
        before anyone else. All of it is kept in a free, downloadable <a href="#data">archive</a>.
      </p>

      <h3>2. The forecast</h3>
      <p>
        For each player and each upcoming match, the model looks at the kind of things an experienced FPL manager
        weighs up:
      </p>
      <ul>
        <li><strong>Form</strong>: points, minutes, goals, assists and xG over the last few matches and the last season or so.</li>
        <li><strong>Role</strong>: does he start, play the full 90, take the big chances? Is he his club's first choice?</li>
        <li><strong>The fixture</strong>: how strong both clubs are, home or away, and how many goals the betting odds expect.</li>
        <li><strong>The crowd</strong>: how many managers are buying or selling him before the deadline, which often
          reflects injury news the stats can't see yet.</li>
        <li><strong>Midweek matches</strong>: whether his club played a cup or European match that week, and how
          many minutes he got in it.</li>
      </ul>
      <p>
        It learned how those things relate to points by studying past seasons.{" "}
        {site.meta.model === "mlp"
          ? <>The model is a small neural network built with a machine-learning library.</>
          : <>The model in use is "{site.meta.model}": {site.meta.model_description}.</>}{" "}
        Its forecast is then adjusted for FPL's injury flags, and double gameweeks count both matches.
      </p>
      <p>
        <strong>Midweek matches.</strong> A club with a Champions League match on Tuesday might rest its stars on
        Saturday. In the data since 2025-26, that doesn't show up: regular starters at clubs in Europe score no less
        after a midweek match than in other weeks. A squad player's midweek role does tell you something. One who
        wasn't used midweek tends to score less than the model expects that weekend, and one who played tends to score
        more. So once the midweek match has been played, the next gameweek's forecast for each player at that club is
        multiplied by:
      </p>
      {groups.length > 0 && (
        <table className="compact">
          <thead><tr><th>Before the weekend</th><th className="num">Forecast ×</th><th className="num">Player-matches</th></tr></thead>
          <tbody>
            {groups.map(([name, g]) => (
              <tr key={name}>
                <td>{ROTATION[name] ?? name}</td>
                <td className="num">{dec(g.factor)}</td>
                <td className="num">{int(g.rows)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p>
        A regular starter is one who has averaged 60 minutes or more over his last five matches.{" "}
        {rotation && <>The numbers come from {rotation.seasons.join(" and ")}, and are pulled towards 1 where there are
          few matches to go on.</>}{" "}
        Later gameweeks aren't adjusted, but the <a href="#next">Next Gameweek</a> page marks each club's midweek matches.
      </p>

      <h3>3. Picking the team</h3>
      <p>
        A good forecast isn't enough on its own: the team still has to fit FPL's rules. An <strong>optimiser</strong> takes
        every player's forecast for the next three gameweeks and searches every legal squad for the one worth the most
        points: £100m budget, two keepers, five defenders, five midfielders and three forwards, no more than three
        from one club, and a valid formation. Along the way it decides:
      </p>
      <ul>
        <li><strong>Transfers</strong>: whether a move beats keeping the squad, or whether to save the free transfer for later.</li>
        <li><strong>Captain</strong>, the starting eleven and the bench order.</li>
        <li><strong>Chips</strong>: whether this week's gain from a Wildcard, Free Hit, Triple Captain or Bench Boost is big enough.</li>
      </ul>
      <p>
        Weeks further ahead count for a little less, because forecasts get less reliable the further out they go.
      </p>

      <h3>4. Checking it</h3>
      <p>
        <strong>Scoring the forecasts.</strong> The model was trained on {trainedOn ?? "past seasons"} and then tested on
        {" "}{comparison?.season ?? "the next season"}, a season it had never seen.
        {ours !== undefined && (
          <> For players who got on the pitch, its forecasts were off by <strong>{dec(ours)} points</strong> per match on
          average.{baseline !== undefined && <> A simple "average of his last five matches" was off by {dec(baseline)}</>}
          {fpl !== undefined && <>, and FPL's own expected points by {dec(fpl)}</>}.</>
        )}
        {ceiling !== undefined && (
          <> That sounds like a lot, but football is noisy: even a "perfect" model that knew each player's true chances
          would be off by about {dec(ceiling)}, because a deflected goal or a missed penalty can't be predicted.</>
        )}{" "}
        This season, every forecast is saved before each deadline and scored once the results are in.
      </p>
      <p>
        <strong>Backtesting.</strong> To test the team picking, the whole system replays past seasons one deadline at a
        time, seeing only what was known before each deadline. It picks a team, makes transfers, scores the real points
        and moves on to the next week.
        {replayModel !== undefined && replaySeasons && <> Replaying {replaySeasons.length} seasons ({replaySeasons[0]} to{" "}
          {replaySeasons[replaySeasons.length - 1]}), each with a model trained only on the seasons before it, it averaged
          about <strong>{int(replayModel)} points a season</strong>
          {replayBaseline !== undefined && <>, against {int(replayBaseline)} picking by the average of each player's last five matches</>}.</>}
        {" "}The settings, such as how many weeks to look ahead and how much a free transfer is worth, were tested the same
        way. They matter much less than the forecasts
        {noise !== undefined && <>: a replayed season moves by about {int(noise)} points on luck alone, more than most of
          them are worth</>}.
      </p>

      <h3>What's on this site</h3>
      <ul>
        <li><a href="#accuracy">Model Accuracy</a>: the full test results.</li>
        <li><a href="#gameweeks">Past Gameweeks</a>: every result, the cup and European matches before it, and each
          player's points against their forecast.</li>
        <li><a href="#next">Next Gameweek</a>: the forecast for the coming gameweek, captain picks, and each club's
          fixtures and midweek matches.</li>
        <li><a href="#players">Players</a>: every player's season, week by week.</li>
        <li><a href="#markets">Markets</a>: what the betting odds said before each deadline, next to what happened. Upcoming
          matches' odds are refreshed regularly.</li>
        <li><a href="#model-team">The Model's Team</a>: a paper FPL team that does whatever the model says, decided before
          every deadline and scored like any other team.</li>
        <li><a href="#data">Data</a>: the archive behind all of it, free to download.</li>
      </ul>

      <p className="note">
        This is a learning project, not advice: check the team news yourself before any deadline. It isn't affiliated
        with the Premier League, Fantasy Premier League or Polymarket.
        {site.meta.repo && <> The code is <a href={site.meta.repo}>on GitHub</a>.</>}
      </p>
    </article>
  );
}
