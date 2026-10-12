import numpy as np
import pandas as pd
import pytest

from xpfpl import field, simulate

# A squad of 15 (2 GKP, 5 DEF, 5 MID, 3 FWD) plus player 99, a popular forward the team doesn't own.
POSITION = {1: 1, 2: 1, **{e: 2 for e in range(3, 8)}, **{e: 3 for e in range(8, 13)}, 13: 4, 14: 4, 15: 4, 99: 4}
TEAM = {"lineup": [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14], "bench": [2, 7, 12, 15], "captain": 14, "vice": 13}


def players(own: dict[int, float] | None = None) -> pd.DataFrame:
    ids = list(POSITION)
    own = own or {}
    return pd.DataFrame({"position": [POSITION[e] for e in ids], "selected_by": [own.get(e, 1.0) for e in ids],
                         "xp_7": [6.0 if e == 99 else 3.0 for e in ids], "p_play": 0.95, "name": [str(e) for e in ids]},
                        index=pd.Index(ids, name="element"))


def test_effective_ownership_shares():
    rng = np.random.default_rng(0)
    n = 600
    t = pd.DataFrame({"selected_by": rng.dirichlet(np.ones(n)) * 1500, "xp_7": rng.uniform(0, 6, n),
                      "p_play": rng.uniform(0.3, 1, n)})
    eo = field.effective_ownership(t, 7)
    assert eo["own"].sum() == pytest.approx(15, rel=1e-6)
    assert eo["start"].sum() == pytest.approx(11, abs=1e-3)           # the average manager fields 11
    assert (eo["start"] <= eo["own"] + 1e-12).all()                    # never more starts than owners
    assert eo["captain"].sum() == pytest.approx(1)                      # one armband each
    assert (eo["eo"] == eo["start"] + eo["captain"]).all()


def test_captaincy_leans_to_the_popular_premium():
    # 74% owned on 5.3 xP against 38% owned on 5.9: squaring ownership gives the armband to the popular one.
    t = pd.DataFrame({"selected_by": [74, 38, 20], "xp_7": [5.3, 5.9, 4.0]})
    cap = field.effective_ownership(t, 7)["captain"]
    assert cap.idxmax() == 0


def test_against_names_threats_and_differentials():
    p = players({99: 80.0, 14: 0.5})                       # 99: owned by most managers, not by the team
    rng = np.random.default_rng(1)
    elements = np.array(sorted(POSITION))
    pts = rng.integers(0, 6, size=(1, 400, len(elements))).astype(np.int8)
    pts[0, :, list(elements).index(99)] = rng.choice([2, 2, 2, 15], size=400)   # a haul one week in four
    d = simulate.Draws([7], elements, pts)
    out = field.against(d, 7, TEAM, p)
    assert 0 <= out["p_beat"] <= 1
    assert out["margin"] == pytest.approx(out["team"] - out["field"], abs=0.02)
    assert out["threats"][0]["element"] == 99 and out["threats"][0]["share"] == 0
    assert any(r["element"] == 14 and r["share"] == 2 for r in out["differentials"])   # a captain few others have
    assert field.against(d, 8, TEAM, p) is None                                         # no simulations for GW8
