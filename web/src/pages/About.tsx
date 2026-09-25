import type { Accuracy } from "../data";
import { dec, int } from "../format";
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
  const tuning = report?.tuning;
  const replay = tuning?.trials?.length ? Math.max(...tuning.trials.map((t: { mean_points: number }) => t.mean_points)) : undefined;
  const trainedOn = report?.validation?.trained_on as string | undefined;

  return (
    <article className="about">
      <h2>What is this?</h2>
      <p className="lede">
        xP-FPL is a personal project that tries to predict how many points every Fantasy Premier League player will
        score, and uses those predictions to pick a team each week. This site is its public record: every gameweek of
        the {site.meta.season} season, what the model expected before each deadline, and what actually happened.
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
      </ul>
      <p>
        It learned how those things relate to points by studying past seasons.{" "}
        {site.meta.model === "mlp"
          ? <>The model is a small neural network built with a machine-learning library.</>
          : <>The model in use is "{site.meta.model}": {site.meta.model_description}.</>}{" "}
        Its forecast is then adjusted for FPL's injury flags, and double gameweeks count both matches.
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
        and moves on to the next week. The settings, such as how many weeks to look ahead and how much a free transfer
        is worth, were chosen by trying different values and keeping whichever scored the most points
        {tuning?.seasons ? ` over ${tuning.seasons.join(" and ")}` : ""}
        {replay !== undefined && <>: about <strong>{int(replay)} points a season</strong></>}.
      </p>

      <h3>What's on this site</h3>
      <ul>
        <li><a href="#accuracy">Model Accuracy</a>: the full test results.</li>
        <li><a href="#gameweeks">Gameweeks</a>: every result, and each player's points against their forecast.</li>
        <li><a href="#players">Players</a>: every player's season, week by week.</li>
        <li><a href="#markets">Markets</a>: what the betting odds said before each deadline, next to what happened. Upcoming
          matches' odds are refreshed regularly.</li>
        <li><a href="#team">My Team</a>: one FPL team's season, each week against the best team it could have picked.</li>
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
