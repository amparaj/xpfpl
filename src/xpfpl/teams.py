"""Team attack and defence ratings: how many goals each side should score in a given fixture.

Rolling "goals for/against over 10 matches" ignores who those goals came against. This fits
the classic Poisson team-strength model (Maher 1982) instead:

    goals(home) ~ Poisson(exp(mu + home + attack[home] - defence[away]))
    goals(away) ~ Poisson(exp(mu +        attack[away] - defence[home]))

It is refitted before every gameweek on the matches played so far, weighting older matches
down with a half-life, so each rating only uses what was known at that deadline. From xG-era
seasons the target is a blend of goals and xG (Poisson likelihoods are happy with non-integer
targets), because xG is far less noisy over a few matches than goals are.

Each fit is a tiny PyTorch problem (~90 parameters, L-BFGS) warm-started from the previous
gameweek's ratings, so ~420 fits over ten seasons take well under a minute. The result is
cached in data/processed/team_ratings.parquet, keyed on the match file's contents.

Features derived from it, per fixture: expected goals for and against, and the Poisson
clean-sheet probability exp(-expected goals against).
"""

import hashlib

import numpy as np
import pandas as pd
import torch

from xpfpl import config

HALF_LIFE_DAYS = 240          # a match this old counts half as much as yesterday's
WINDOW_DAYS = 3 * 365         # older than this is ignored entirely
XG_BLEND = 0.5                # target = (1 - XG_BLEND) * goals + XG_BLEND * xG, when xG exists
L2 = 2.0                      # pull towards the prior; matters for teams with little data
PROMOTED_PRIOR = (-0.25, -0.25)   # a club with little recent top-flight data: weaker at both ends
MIN_WEIGHT = 10.0             # weighted matches below which a club counts as "new"
NEUTRAL = (0.25, 0.2)         # (mu, home) when there are no ratings at all: ~1.3 and ~1.6 goals
CACHE_PATH = config.PROCESSED_DIR / "team_ratings.parquet"


def fixtures_table(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture: home/away club codes, goals and (where recorded) xG."""
    m = matches[["season", "gw", "fixture", "kickoff_time", "team_code", "was_home",
                 "team_h_score", "team_a_score", "expected_goals"]]
    xg = m.groupby(["season", "fixture", "was_home"])["expected_goals"].sum(min_count=1).unstack()
    first = m.groupby(["season", "fixture", "was_home"])["team_code"].first().unstack()
    meta = m.groupby(["season", "fixture"]).agg(gw=("gw", "first"), kickoff_time=("kickoff_time", "first"),
                                                 hg=("team_h_score", "first"), ag=("team_a_score", "first"))
    out = meta.assign(home=first[True], away=first[False], hxg=xg[True], axg=xg[False]).reset_index()
    return out.dropna(subset=["home", "away", "hg", "ag"]).sort_values("kickoff_time").reset_index(drop=True)


class _Ratings(torch.nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor(0.3))
        self.home = torch.nn.Parameter(torch.tensor(0.2))
        self.attack = torch.nn.Parameter(torch.zeros(n))
        self.defence = torch.nn.Parameter(torch.zeros(n))

    def rates(self, h: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        home = torch.exp(self.mu + self.home + self.attack[h] - self.defence[a])
        away = torch.exp(self.mu + self.attack[a] - self.defence[h])
        return home, away


def _fit(model: _Ratings, h, a, hg, ag, w, prior_att, prior_def) -> None:
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        lh, la = model.rates(h, a)
        nll = (w * (lh - hg * torch.log(lh))).sum() + (w * (la - ag * torch.log(la))).sum()
        penalty = L2 * (((model.attack - prior_att) ** 2).sum() + ((model.defence - prior_def) ** 2).sum())
        loss = (nll + penalty) / w.sum()
        loss.backward()
        return loss

    opt.step(closure)


def _fit_at(fx: pd.DataFrame, moments: list[tuple[str, int, pd.Timestamp]]) -> pd.DataFrame:
    """Ratings as known at each (season, gw, deadline) in `moments`, which must be in time order."""
    codes = sorted(set(fx["home"]) | set(fx["away"]))
    index = {c: i for i, c in enumerate(codes)}
    h_all = torch.tensor(fx["home"].map(index).to_numpy())
    a_all = torch.tensor(fx["away"].map(index).to_numpy())

    def blend(g, xg):
        return np.where(np.isnan(xg), g, (1 - XG_BLEND) * g + XG_BLEND * np.nan_to_num(xg))

    hg_all = torch.tensor(blend(fx["hg"].to_numpy(float), fx["hxg"].to_numpy(float)), dtype=torch.float32)
    ag_all = torch.tensor(blend(fx["ag"].to_numpy(float), fx["axg"].to_numpy(float)), dtype=torch.float32)
    kickoff = fx["kickoff_time"]

    model = _Ratings(len(codes))          # warm-started from one gameweek to the next
    rows = []
    for season, gw, start in moments:
        age = (start - kickoff).dt.total_seconds().to_numpy() / 86400.0
        known = (age > 0) & (age < WINDOW_DAYS)
        if known.sum() < 20:
            continue
        idx = torch.tensor(np.flatnonzero(known))
        w = torch.tensor(0.5 ** (age[known] / HALF_LIFE_DAYS), dtype=torch.float32)
        h, a = h_all[idx], a_all[idx]
        exposure = torch.zeros(len(codes)).index_add_(0, h, w).index_add_(0, a, w)
        new = exposure < MIN_WEIGHT
        prior_att = torch.where(new, PROMOTED_PRIOR[0], 0.0)
        prior_def = torch.where(new, PROMOTED_PRIOR[1], 0.0)
        _fit(model, h, a, hg_all[idx], ag_all[idx], w, prior_att, prior_def)
        with torch.no_grad():
            rows.append(pd.DataFrame({"season": season, "gw": gw, "team_code": codes,
                                      "attack": model.attack.numpy().copy(),
                                      "defence": model.defence.numpy().copy(),
                                      "mu": float(model.mu), "home": float(model.home)}))
    return pd.concat(rows, ignore_index=True)


def fit_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Ratings as known just before each (season, gw): one row per club per gameweek."""
    fx = fixtures_table(matches)
    starts = fx.groupby(["season", "gw"])["kickoff_time"].min().sort_values()
    return _fit_at(fx, [(s, g, t) for (s, g), t in starts.items()])


def _fingerprint(matches: pd.DataFrame) -> str:
    key = f"{len(matches)}|{matches['kickoff_time'].max()}|{HALF_LIFE_DAYS}|{WINDOW_DAYS}|{XG_BLEND}|{L2}"
    return hashlib.md5(key.encode()).hexdigest()


def ratings(matches: pd.DataFrame) -> pd.DataFrame:
    """`fit_history`, cached on disk until the match data (or a setting above) changes."""
    stamp = _fingerprint(matches)
    if CACHE_PATH.exists():
        cached = pd.read_parquet(CACHE_PATH)
        if len(cached) and cached["stamp"].iloc[0] == stamp:
            return cached.drop(columns="stamp")
    table = fit_history(matches)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.assign(stamp=stamp).to_parquet(CACHE_PATH, index=False)
    return table


def latest(matches: pd.DataFrame, season: str, gw: int) -> pd.DataFrame:
    """Ratings after every match played so far, for `season`'s upcoming `gw`: one row per club."""
    fx = fixtures_table(matches)
    moment = fx["kickoff_time"].max() + pd.Timedelta(hours=1)
    return _fit_at(fx, [(season, gw, moment)]).set_index("team_code")


def fixture_rates(table: pd.DataFrame, team_code, opp_code, was_home) -> tuple[np.ndarray, np.ndarray]:
    """Expected goals for and against for each (team, opponent, home?) given one ratings table
    (one row per club, indexed by team_code). Unknown clubs get the promoted-club prior."""
    att = table["attack"]
    dfn = table["defence"]
    mu, home = (float(table["mu"].iloc[0]), float(table["home"].iloc[0])) if len(table) else NEUTRAL
    t = pd.Series(np.asarray(team_code))
    o = pd.Series(np.asarray(opp_code))
    ta, td = t.map(att).fillna(PROMOTED_PRIOR[0]), t.map(dfn).fillna(PROMOTED_PRIOR[1])
    oa, od = o.map(att).fillna(PROMOTED_PRIOR[0]), o.map(dfn).fillna(PROMOTED_PRIOR[1])
    h = np.asarray(was_home, dtype=float)
    gf = np.exp(mu + home * h + ta.to_numpy() - od.to_numpy())
    ga = np.exp(mu + home * (1 - h) + oa.to_numpy() - td.to_numpy())
    return gf, ga
