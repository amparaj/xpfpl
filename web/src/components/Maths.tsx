// The Technical Report's maths section: every formula behind a forecast and a pick, each said in
// words first, then written out, then worked through with FPL numbers. Formulas are TeX turned
// into MathML by KaTeX, which the browser draws itself (no fonts or stylesheet to load). Loaded
// lazily by Report.tsx so KaTeX stays out of the main bundle.

import katex from "katex";
import { useMemo, type ReactNode } from "react";
import { dec, pct } from "../format";

/** A formula: inline by default, on its own line with `block`. */
export function M({ t, block }: { t: string; block?: boolean }) {
  const html = useMemo(() => katex.renderToString(t, { output: "mathml", displayMode: !!block, throwOnError: false }), [t, block]);
  return block
    ? <div className="maths-block" dangerouslySetInnerHTML={{ __html: html }} />
    : <span className="maths-inline" dangerouslySetInnerHTML={{ __html: html }} />;
}

function Part({ id, title, sub, children }: { id: string; title: string; sub: string; children: ReactNode }) {
  return (
    <details className="maths-part" id={`maths-${id}`}>
      <summary><strong>{title}</strong><span className="muted">{sub}</span></summary>
      <div className="maths-body">{children}</div>
    </details>
  );
}

function Example({ children }: { children: ReactNode }) {
  return <div className="maths-example"><span className="maths-example-label">Example</span>{children}</div>;
}

export interface MathsProps {
  horizon: number;
  discount: number;
  bench: number;
  ftValue: number;
  priceWeight: number;
  outThreshold?: number;
  calibration?: { intercept: number; slope: number };
}

export default function Maths({ horizon, discount, bench, ftValue, priceWeight, outThreshold, calibration }: MathsProps) {
  const weights = Array.from({ length: horizon }, (_, k) => dec(discount ** k, 2)).join(", ");
  return (
    <>
      <p>
        The formulas behind each step, in the order a forecast is made. Each part says in words what the formula does, writes it
        out, then works an example with FPL numbers. Click a heading to open it. Notation:{" "}
        <M t={String.raw`\mathbb{E}[X]`} /> is the expected (average) value of <M t="X" />, <M t={String.raw`P(\cdot)`} /> a
        chance, <M t="i" /> a player, <M t="g" /> a gameweek, and a hat (<M t={String.raw`\hat{y}`} />) marks a forecast
        of <M t="y" />.
      </p>

      <Part id="inputs" title="1. Turning matches into inputs" sub="rolling averages, rates per 90, standardising">
        <p>
          Recent form is the average over a player's last <M t="n" /> matches (3, 5 or 10), counting only matches played before
          the one being forecast:
        </p>
        <M block t={String.raw`\bar{x}_n = \frac{1}{n}\sum_{k=1}^{n} x_{t-k}`} />
        <p>
          His underlying level is a rate per 90 minutes over his last 20 or 38 matches. Dividing by minutes stops a 20-minute
          cameo dragging the rate down; at least 90 minutes go on the bottom so a single cameo can't make a huge rate:
        </p>
        <M block t={String.raw`\text{rate}_{90} = 90 \times \frac{\sum \text{stat}}{\max\!\left(\sum \text{minutes},\ 90\right)}`} />
        <p>
          The neural networks then need every input on the same scale, so each is standardised with its mean <M t={String.raw`\mu`} /> and
          standard deviation <M t={String.raw`\sigma`} /> in the training rows (an input that never varied in training gets <M t={String.raw`\sigma = 1`} />):
        </p>
        <M block t={String.raw`z = \frac{x - \mu}{\sigma}`} />
        <Example>3 goals in 1,620 minutes over his last 20 matches is <M t={String.raw`90 \times 3 / 1620 = 0.17`} /> goals a match.</Example>
      </Part>

      <Part id="ratings" title="2. Club ratings" sub="a Poisson model of goals, refitted every gameweek">
        <p>
          Goals are rare and roughly independent, so the number a side scores is modelled as Poisson: if a side expects <M t={String.raw`\lambda`} /> goals,
          the chance it scores exactly <M t="k" /> is
        </p>
        <M block t={String.raw`P(k) = \frac{\lambda^k e^{-\lambda}}{k!}`} />
        <p>
          Each club has an attack rating <M t="a" /> and a defence rating <M t="d" />; <M t={String.raw`\mu`} /> sets the league's scoring level
          and <M t="h" /> is home advantage (Maher, 1982):
        </p>
        <M block t={String.raw`\lambda_{\text{home}} = e^{\,\mu + h + a_{\text{home}} - d_{\text{away}}}`} />
        <M block t={String.raw`\lambda_{\text{away}} = e^{\,\mu + a_{\text{away}} - d_{\text{home}}}`} />
        <p>
          The ratings are the ones that make the matches played so far most likely. The target <M t="y" /> is half goals and half xG
          (xG is far steadier over a few matches). Each match is weighted by its age, halving every 240 days, and the ratings are
          pulled gently towards 0 (towards −0.25 for a club with little recent top-flight data, such as a promoted one):
        </p>
        <M block t={String.raw`\begin{aligned} \min_{\mu,h,a,d}\ & \sum_{m} w_m \big(\lambda_m - y_m \ln \lambda_m\big) \\ & + 2\sum_{c}\Big[(a_c - a^0_c)^2 + (d_c - d^0_c)^2\Big] \end{aligned}`} />
        <M block t={String.raw`w_m = 0.5^{\,\text{age}_m / 240}`} />
        <p>
          The first sum is the Poisson log-likelihood turned into a loss (minus its log, without the parts that don't depend on
          the ratings), over both sides of every match; the second is the pull towards the prior. PyTorch minimises it with
          L-BFGS. A side's clean-sheet chance is then the Poisson chance of conceding none:
        </p>
        <M block t={String.raw`P(\text{clean sheet}) = P(0) = e^{-\lambda_{\text{against}}}`} />
        <Example>
          With <M t={String.raw`\mu = 0.25`} />, <M t="h = 0.2" />, a home attack of 0.3 and an away defence of 0.1, the home side
          expects <M t={String.raw`e^{0.65} = 1.92`} /> goals. A side expected to concede 1.2 keeps a clean sheet <M t={String.raw`e^{-1.2} = 30\%`} /> of the time.
        </Example>
      </Part>

      <Part id="markets" title="3. Betting odds" sub="fair prices, and goal rates fitted to every market at once">
        <p>
          Polymarket's home, draw and away prices <M t="q" /> are scaled to add up to 1 (they are dropped if they are more than 0.1 out,
          a sign of a stale price):
        </p>
        <M block t={String.raw`p_k = \frac{q_k}{q_{\text{home}} + q_{\text{draw}} + q_{\text{away}}}`} />
        <p>
          Then each side's expected goals are found from all of the match's markets together. With independent Poisson goals the
          chance of every scoreline is <M t={String.raw`P(i, j) = P(i \mid \lambda_h)\,P(j \mid \lambda_a)`} />, and every market is a sum over scorelines:
        </p>
        <M block t={String.raw`\begin{aligned}
P(\text{home win}) &= \textstyle\sum_{i > j} P(i,j) \\ P(\text{draw}) &= \textstyle\sum_{i = j} P(i,j) \\
P(\text{over } 2.5) &= \textstyle\sum_{i + j > 2.5} P(i,j) \\ P(\text{both score}) &= (1 - e^{-\lambda_h})(1 - e^{-\lambda_a})
\end{aligned}`} />
        <p>
          The two rates are chosen by gradient descent (Adam, in PyTorch, every match at once) to make these as close as possible
          to the prices, over whichever markets exist for that match:
        </p>
        <M block t={String.raw`\min_{\lambda_h, \lambda_a} \sum_{\text{markets}} \big(P_{\text{implied}} - p_{\text{market}}\big)^2`} />
        <p>
          These rates replace the club ratings' expected goals wherever a match is priced. The component model also moves its own
          clean-sheet chance halfway to the market's:{" "}
          <M t={String.raw`P(\text{CS}) = \tfrac{1}{2}P_{\text{model}} + \tfrac{1}{2}\,P(60{+})\,e^{-\lambda_{\text{against}}}`} />.
        </p>
        <Example>Prices of 0.52, 0.27 and 0.23 add up to 1.02, so the fair chances are 0.510, 0.265 and 0.225.</Example>
      </Part>

      <Part id="rules" title="4. FPL's rules as a formula" sub="why expected points add up, and the one place they don't">
        <p>FPL's scoring is a sum of parts. For one player in one match:</p>
        <M block t={String.raw`\begin{aligned} \text{pts} = {} & \underbrace{A_1 + A_{60}}_{\text{appearance}} + G_{\text{pos}}\,\text{goals} + 3\,\text{assists} \\ & + C_{\text{pos}}\,\text{CS} + \Big\lfloor \tfrac{\text{saves}}{3} \Big\rfloor - \Big\lfloor \tfrac{\text{conceded}}{2} \Big\rfloor \\ & + \text{bonus} + 2\,\text{DC} + \text{other} \end{aligned}`} />
        <p>
          <M t="A_1" /> and <M t="A_{60}" /> are 1 for playing and 1 more for 60+ minutes; <M t={String.raw`G_{\text{pos}}`} /> is 10, 6, 5 or 4 a goal and{" "}
          <M t={String.raw`C_{\text{pos}}`} /> 4, 4, 1 or 0 a clean sheet (GKP, DEF, MID, FWD); saves count for goalkeepers only and goals conceded
          for goalkeepers and defenders who played 60+; DC is 1 when the defensive-contribution target is reached; "other" is cards,
          own goals and penalties. Because an average of a sum is the sum of the averages, expected points are the same formula
          with each part replaced by its expectation:
        </p>
        <M block t={String.raw`\begin{aligned} \mathbb{E}[\text{pts}] = {} & P(\text{plays}) + P(60{+}) \\ & + G_{\text{pos}}\,\mathbb{E}[\text{goals}] \\ & + 3\,\mathbb{E}[\text{assists}] + \dots \end{aligned}`} />
        <p>
          The exception is a rule paying a point per 2 or per 3 of something: rounding down doesn't pass through an average. Its
          expectation is taken over a Poisson count <M t="X" /> instead:
        </p>
        <M block t={String.raw`\mathbb{E}\Big[\Big\lfloor \tfrac{X}{n} \Big\rfloor\Big] = \sum_{k=0}^{\infty} \Big\lfloor \tfrac{k}{n} \Big\rfloor P(X = k)`} />
        <Example>
          A defender expected to concede 1.3 loses <M t="0.42" /> points on average, not <M t="1.3 / 2 = 0.65" />: conceding one costs nothing.
          A keeper expected to make 3 saves gets <M t="0.66" /> points, not 1.
        </Example>
      </Part>

      <Part id="network" title="5. The neural network" sub="layers, the loss, and how it learns">
        <p>
          A forecast is a chain of simple steps. The standardised inputs <M t="z" /> are multiplied by a table of weights <M t="W" />,
          shifted by <M t="b" /> and passed through ReLU, which keeps positive values and zeroes negative ones. Two such layers
          (128 then 64 numbers wide) feed one output, the xP:
        </p>
        <M block t={String.raw`\begin{aligned} h_1 &= \text{ReLU}(W_1 z + b_1) \\ h_2 &= \text{ReLU}(W_2 h_1 + b_2) \\ \hat{y} &= w_3^{\top} h_2 + b_3 \end{aligned}`} />
        <M block t={String.raw`\text{ReLU}(u) = \max(0, u)`} />
        <p>
          Training chooses the weights that make the mean squared error over past player-matches as small as possible. Squared error
          is used because the forecast that minimises it is the average outcome: exactly what expected points are, hauls included.
        </p>
        <M block t={String.raw`L = \frac{1}{N}\sum_{r=1}^{N} \big(\hat{y}_r - y_r\big)^2`} />
        <p>
          The weights move a small step downhill on the loss after every batch of 1,024 rows. The plain version is{" "}
          <M t={String.raw`\theta \leftarrow \theta - \eta\,\nabla L`} />; Adam, used here, keeps running averages of each weight's gradient{" "}
          <M t="g" /> and its square, so each weight gets its own step size:
        </p>
        <M block t={String.raw`\begin{aligned} m &\leftarrow \beta_1 m + (1-\beta_1)\,g \\ v &\leftarrow \beta_2 v + (1-\beta_2)\,g^2 \\ \theta &\leftarrow \theta - \eta\,\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} \end{aligned}`} />
        <p>
          with <M t={String.raw`\eta = 0.001`} />, <M t={String.raw`\beta_1 = 0.9`} />, <M t={String.raw`\beta_2 = 0.999`} /> (<M t={String.raw`\hat{m}, \hat{v}`} /> correct the
          averages' slow start). Two guards against learning the training rows by heart: dropout switches off a random 10% of the
          layer values at each step, and early stopping ends training once the error on a held-out season hasn't improved for 5
          passes over the data.
        </p>
      </Part>

      <Part id="minutes" title="6. The minutes model" sub="chances of 0, 1-59 and 60+ minutes, and points given each">
        <p>
          The same kind of network, but its last layer gives three scores <M t="s_k" />, one per minutes class, turned into chances
          that add up to 1 by the softmax:
        </p>
        <M block t={String.raw`p_k = \frac{e^{s_k}}{e^{s_{\text{none}}} + e^{s_{\text{cameo}}} + e^{s_{\text{60+}}}}`} />
        <p>
          Two more outputs give the points expected in a cameo, <M t={String.raw`\mu_{\text{cameo}}`} />, and in a 60+ game, <M t={String.raw`\mu_{60}`} />. No minutes
          scores nothing, so
        </p>
        <M block t={String.raw`\text{xP} = p_{\text{cameo}}\,\mu_{\text{cameo}} + p_{60}\,\mu_{60}`} />
        <p>
          It is trained on three losses added together: cross-entropy for the class (minus the log of the chance it gave to what
          happened, so a confident miss costs a lot), and squared errors for the points (weighted 0.25) and minutes, each learned
          only from the rows in its class:
        </p>
        <M block t={String.raw`\begin{aligned} L = {} & -\ln p_{\text{actual class}} \\ & + 0.25\,(\hat{\mu} - \text{pts})^2 + \big(\hat{m} - \tfrac{\text{minutes}}{90}\big)^2 \end{aligned}`} />
        <Example>
          Scores of 0.5, 1.0 and 2.0 give chances of 14%, 23% and 63%. With 1.5 points expected from a cameo and 5.5 from 60+,
          xP is <M t={String.raw`0.23 \times 1.5 + 0.63 \times 5.5 = 3.8`} />.
        </Example>
      </Part>

      <Part id="boosting" title="7. Gradient-boosted trees" sub="many small decision trees, each fixing the last one's errors">
        <p>
          Start from the average score. Each round fits a decision tree (up to 63 leaves) to what's still wrong, the residuals{" "}
          <M t={String.raw`r = y - F_{m-1}(x)`} />, and adds a small part of it:
        </p>
        <M block t={String.raw`F_m(x) = F_{m-1}(x) + \eta\, f_m(x), \qquad \eta = 0.05`} />
        <p>
          For squared error the residuals are the downhill direction of the loss, hence "gradient" boosting. Rounds stop once the
          held-out season hasn't improved for 50. Because the forecast is a sum, it can be split into each input's share (SHAP
          values, used for "What goes with points?"):
        </p>
        <M block t={String.raw`F(x) = \phi_0 + \sum_{j} \phi_j(x)`} />
      </Part>

      <Part id="ensemble" title="8. The ensemble and its confidence" sub="an average of three, and how far they disagree">
        <M block t={String.raw`\text{xP} = \tfrac{1}{3}\big(\text{xP}_{\text{net}} + \text{xP}_{\text{trees}} + \text{xP}_{\text{mins}}\big)`} />
        <p>
          Confidence is the three forecasts' standard deviation <M t="s" /> over the xP (at least 1, so tiny forecasts don't look
          unsure), cut at 0.09 and 0.164:
        </p>
        <M block t={String.raw`c = \frac{s}{\max(\text{xP}, 1)}`} />
        <M block t={String.raw`\text{confidence} = \begin{cases} \text{High} & c \le 0.09 \\ \text{Medium} & 0.09 < c \le 0.164 \\ \text{Low} & c > 0.164 \end{cases}`} />
        <Example>
          Forecasts of 4.6, 5.2 and 4.3 average 4.7 with <M t="s = 0.46" />, so <M t={String.raw`c = 0.46 / 4.7 = 0.098`} />: Medium.
        </Example>
      </Part>

      <Part id="after" title="9. After the forecast" sub="availability, midweek matches, the betting market, doubles">
        <p>Each fixture's xP is scaled by the player's chance of being available, the midweek factor and the betting market:</p>
        <M block t={String.raw`\text{xP}_{\text{final}} = \text{xP} \times p_{\text{avail}} \times f_{\text{midweek}} \times m`} />
        <M block t={String.raw`m = \begin{cases} 0.1 & \text{scorer odds} \le ${outThreshold != null ? `${Math.round(outThreshold * 100)}\\%` : "\\text{threshold}"} \\ 1 & \text{otherwise} \end{cases}`} />
        <p>
          Next gameweek, <M t={String.raw`p_{\text{avail}}`} /> is FPL's chance of playing (or the press conference's: out 0, doubtful 0.5, fit 1). For
          the weeks after, a flagged player recovers 25% a week, unless FPL gives a return date (a ban then counts as fully available,
          an injury at least 75%):
        </p>
        <M block t={String.raw`p_{\text{avail}}(k) = \min\!\big(1,\ p_0 + 0.25\,k\big)`} />
        <p>for the <M t="k" />th gameweek after the next (<M t="k = 0" /> is next week, <M t="p_0" /> its chance).</p>
        <p>A double gameweek adds its two matches; a blank scores 0.</p>
        <Example>A player flagged 50% gets 0.5 next week, then 0.75, then 1.</Example>
      </Part>

      <Part id="simulation" title="10. The Monte Carlo simulation" sub="thousands of gameweeks, built from the rules">
        <p>Each simulated gameweek, for every fixture:</p>
        <ol>
          <li>Each side's goals: <M t={String.raw`G \sim \text{Poisson}(\lambda)`} />, from the market or the club ratings.</li>
          <li>
            The line-up, with each player picked with his chance <M t="p_i" /> of 60+ minutes (then the same again for cameos). Players are
            shuffled and their chances laid end to end; with one random <M t={String.raw`u`} /> between 0 and 1 for the side, player <M t="i" /> is
            picked if a whole number falls in his stretch:
            <M block t={String.raw`C_i = \textstyle\sum_{j \le i} p_j`} />
            <M block t={String.raw`\text{picked}_i \iff \lfloor C_i + u \rfloor > \lfloor C_{i-1} + u \rfloor`} />
            Each player keeps exactly his own chance, and a side always fields within one of the number its chances add up to.
          </li>
          <li>
            Each of the side's goals is his with chance <M t={String.raw`\pi_i`} />: his rate per 90 times the share of the match he's on the pitch
            (<M t={String.raw`\tau`} /> = 0.96 for 60+, 0.25 for a cameo) over the side's expected goals. Assists likewise.
            <M block t={String.raw`\pi_i = \frac{\text{rate}_i \, \tau_i}{\lambda}`} />
            <M block t={String.raw`\text{goals}_i \sim \text{Binomial}\big(G,\ \pi_i\big)`} />
            So teammates share one set of goals, and the club's total is still its Poisson draw.
          </li>
          <li>
            A clean sheet if the other side scored none (60+ minutes); saves <M t={String.raw`\sim \text{Poisson}(\text{rate}\cdot\tau)`} />;
            defensive actions <M t={String.raw`\sim \text{Poisson}(\text{rate}\cdot\tau\cdot M)`} /> with <M t={String.raw`M \sim \text{Gamma}(16, \tfrac{1}{16})`} /> (mean 1,
            a little extra spread), scoring 2 at 10 (DEF) or 12 (MID, FWD).
          </li>
          <li>Bonus from how often players with the same position, goals, assists and clean sheet got 0, 1, 2 or 3 bonus in past matches; cards and the like from past matches.</li>
        </ol>
        <p>
          The points then follow from the rules (part 4). Last, each player's average is matched to his xP. The points that don't
          depend on attacking returns (appearance, clean sheet, defence) are <M t="b" />, the attacking ones (goals, assists, bonus, saves) average{" "}
          <M t="a" />; his attacking rates are scaled by
        </p>
        <M block t={String.raw`k = \text{clip}\!\left(\frac{\text{xP} - b}{a},\ 0.25,\ 6\right)`} />
        <p>
          and if even that leaves too much (he's a doubt), his chance of playing comes down by <M t={String.raw`\min\big(1, \text{xP} / (b + k a)\big)`} />. Two
          passes of 1,000 simulations set the scales, then 5,000 are kept. A player's range is the 10th, 50th and 90th percentile of his
          simulated scores, and every chance shown is a share of the simulations:
        </p>
        <M block t={String.raw`P(10{+}) \approx \frac{1}{N}\sum_{s=1}^{N} \mathbf{1}\big[\text{pts}_s \ge 10\big]`} />
        <p>A chance <M t="p" /> counted this way from <M t="N" /> simulations is uncertain by its standard error:</p>
        <M block t={String.raw`\text{SE} = \sqrt{\frac{p(1-p)}{N}}`} />
        <Example>
          A forward scoring 0.6 a match, playing 60+ for a side expected to score 1.8, takes each goal with chance{" "}
          <M t={String.raw`0.6 \times 0.96 / 1.8 = 0.32`} />. If the side scores 2 he gets at least one <M t={String.raw`1 - 0.68^2 = 54\%`} /> of the time. Over
          5,000 simulations a 20% chance is known to within <M t={String.raw`\sqrt{0.2 \times 0.8 / 5000} = 0.6`} /> points of a percent.
        </Example>
      </Part>

      <Part id="scores" title="11. Measuring accuracy" sub="RMSE, MAE, R², rank correlation, log score, RPS">
        <p>For forecasts <M t={String.raw`\hat{y}`} /> of points <M t="y" /> over <M t="N" /> player-matches:</p>
        <M block t={String.raw`\text{RMSE} = \sqrt{\frac{1}{N}\sum (y - \hat{y})^2}`} />
        <M block t={String.raw`\text{MAE} = \frac{1}{N}\sum |y - \hat{y}|`} />
        <M block t={String.raw`R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}`} />
        <p>
          RMSE squares the misses, so a missed haul counts heavily; MAE treats every point of error alike. <M t="R^2" /> is the share of
          the variation in points that the forecasts explain (0 = no better than everyone's average). Rank correlation (Spearman) only
          asks whether the order is right, from the gap <M t="d" /> between each player's forecast rank and actual rank:
        </p>
        <M block t={String.raw`\rho = 1 - \frac{6\sum d^2}{N(N^2 - 1)}`} />
        <p>
          Calibration fits a straight line through points against xP by least squares
          {calibration ? <>: over the walk-forward seasons <M t={String.raw`\text{pts} \approx ${dec(calibration.intercept, 2)} + ${dec(calibration.slope, 2)}\,\text{xP}`} />, close to the ideal 0 + 1 × xP</> : <> (ideally 0 + 1 × xP)</>}.
          A whole spread of chances <M t="p" /> over scores is scored by the log score and the ranked probability score, with <M t="F" /> the
          forecast's running total of chances:
        </p>
        <M block t={String.raw`\text{log score} = -\ln p(y)`} />
        <M block t={String.raw`\text{RPS} = \sum_{k} \big(F(k) - \mathbf{1}[y \le k]\big)^2`} />
        <p>
          Both are lower-is-better. The log score only looks at the chance given to what happened; RPS also rewards being near it.
          Match results are scored the same way: log loss <M t={String.raw`-\ln p_{\text{outcome}}`} /> and the Brier score{" "}
          <M t={String.raw`\sum_k (p_k - o_k)^2`} /> over home, draw and away, with <M t="o_k" /> 1 for what happened. The simulation's ranges are checked by{" "}
          coverage: the share of real scores inside the 10th to 90th percentile should be {pct(0.8)}.
        </p>
        <Example>A forecast giving the score that happened a 25% chance has a log score of <M t={String.raw`-\ln 0.25 = 1.39`} />.</Example>
      </Part>

      <Part id="optimiser" title="12. Picking the team" sub="an integer program over FPL's rules">
        <p>
          Every choice is a yes/no variable: <M t="x_i" /> (in the squad), <M t={String.raw`s_{ig}`} /> (starts in gameweek <M t="g" />), <M t={String.raw`c_{ig}`} /> (captain).
          The solver finds the values that make this as large as possible:
        </p>
        <M block t={String.raw`\begin{aligned} \max \ & \sum_{g} \delta^{\,g - g_0} \sum_i \Big[\text{xP}_{ig}\big(s_{ig} + c_{ig} \\ & \qquad + \beta\,(x_i - s_{ig})\big) + \omega\,\Delta_i\,x_i\Big] \\ & - 4\,\text{hits} + v\,\text{saved} \end{aligned}`} />
        <p>
          The captain's xP counts twice because <M t={String.raw`c_{ig}`} /> adds it again; the bench counts for <M t={String.raw`\beta`} />; <M t={String.raw`\Delta_i`} /> is his expected price
          change (an average of a form-and-price table and this week's net transfers); "saved" is a free transfer kept for next week.
          With the settings chosen by replaying seasons: <M t={String.raw`\delta = ${discount}`} /> over {horizon} gameweeks (weights {weights}),{" "}
          <M t={String.raw`\beta = ${bench}`} />, <M t={String.raw`\omega = ${priceWeight}`} /> points per £m and <M t={String.raw`v = ${ftValue}`} />. Subject to:
        </p>
        <M block t={String.raw`\begin{aligned}
&\textstyle\sum_{\text{GKP}} x_i = 2,\ \ \sum_{\text{DEF}} x_i = 5 \\
&\textstyle\sum_{\text{MID}} x_i = 5,\ \ \sum_{\text{FWD}} x_i = 3 \\
&\textstyle\sum_{i \in \text{club}} x_i \le 3 \text{ for every club} \\
&\textstyle\sum_i s_{ig} = 11,\ \ \sum_i c_{ig} = 1,\ \ c_{ig} \le s_{ig} \le x_i \\
&\text{XI: } 1 \text{ GKP},\ 3\text{–}5 \text{ DEF},\ 2\text{–}5 \text{ MID},\ 1\text{–}3 \text{ FWD} \\
&\textstyle\sum_i \text{price}_i\,x_i \le \text{bank} + \text{selling value} \\
&\text{hits} \ge \text{transfers} - \text{free} \\
&\text{saved} \le \text{free} - \text{transfers}
\end{aligned}`} />
        <p>
          A Triple Captain week counts <M t={String.raw`c_{ig}`} /> twice more, a Bench Boost week sets <M t={String.raw`\beta = 1`} />. The selling price of a player bought
          at <M t="b" /> and now worth <M t="n" /> keeps half of any rise, rounded down to £0.1m, and passes on falls in full:
        </p>
        <M block t={String.raw`\text{sell} = \begin{cases} b + \big\lfloor 10 \cdot \tfrac{n - b}{2} \big\rfloor / 10 & n > b \\ n & n \le b \end{cases}`} />
        <Example>Bought at £7.0m and now £7.5m: sells for £7.2m. A £0.3m rise sells for £7.1m (half of 0.3 is 0.15, rounded down).</Example>
      </Part>

      <Part id="chips" title="13. Chips" sub="a gain against a threshold">
        <p>Each chip is worth the xP it adds over not playing it:</p>
        <M block t={String.raw`\begin{aligned}
\text{Triple Captain:}\ & \text{xP}_{\text{captain}} \ge 6 \\
\text{Bench Boost:}\ & \text{xP}_{\text{bench}} \ge 8 \\
\text{Free Hit:}\ & \text{xP}_{\text{one-week team}} - \text{xP}_{\text{plan}} \ge 18 \\
\text{Wildcard:}\ & \text{value}_{\text{rebuilt}} - \text{value}_{\text{plan}} \ge 30
\end{aligned}`} />
        <p>
          and is held back if a later gameweek in the same chip window looks better. The thresholds were chosen by replaying seasons.
          The simulations add the chance the gain clears the threshold, from the same simulated weeks with and without the chip.
        </p>
      </Part>

      <Part id="testing" title="14. Telling settings apart" sub="paired replays and the bootstrap">
        <p>
          In a paired test the candidate setting takes over the base's exact squad, bank and free transfers at each deadline, plays the
          next <M t="w" /> gameweeks, and is scored against the base over the same weeks. The windows overlap (each week is in{" "}
          <M t="w" /> of them), so each window's gain is divided by <M t="w" /> and a season's gain is the sum over its deadlines:
        </p>
        <M block t={String.raw`\Delta = \sum_{\text{deadlines}} \frac{1}{w}\Big(\text{pts}^{\text{new}}_{w} - \text{pts}^{\text{base}}_{w}\Big)`} />
        <p>
          Its 90% range comes from a moving-block bootstrap: resample runs of <M t="w" /> consecutive deadlines with replacement (runs, because
          neighbouring windows share weeks), recompute <M t={String.raw`\Delta`} /> 2,000 times and take the 5th and 95th percentiles. Comparing
          both settings in the same weeks removes most of the luck that makes two separate replays hard to compare.
        </p>
      </Part>
    </>
  );
}
