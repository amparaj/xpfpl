from xpfpl import chips

BS = {"chips": [{"name": n, "start_event": s, "stop_event": e}
                for n in ("wildcard", "freehit", "3xc", "bboost") for s, e in ((1, 19), (20, 38))]}


def _available(gw, used=()):
    return chips.available_chips(BS, [{"name": n, "event": e} for n, e in used], gw)


def test_no_warning_with_room_to_spare():
    # Three chips left (wildcard played in GW6), GW7: 13 weeks for 3 chips.
    assert chips.window_warning(_available(7, [("wildcard", 6)]), 7) is None
    # Six weeks for three chips is exactly three spare: the first week it warns.
    assert chips.window_warning(_available(13, [("wildcard", 6)]), 13) is None
    msg = chips.window_warning(_available(14, [("wildcard", 6)]), 14)
    assert msg.startswith("3 chips left (Free Hit, Triple Captain, Bench Boost) and 6 GWs (GW14-19)")


def test_warning_gets_stronger_as_weeks_run_out():
    used = [("wildcard", 6)]
    assert "play one every week" in chips.window_warning(_available(17, used), 17)
    msg = chips.window_warning(_available(18, used), 18)
    assert "only 2 GWs (GW18-19)" in msg and "1 will expire unused" in msg
    one = chips.window_warning(_available(19, used + [("freehit", 10), ("3xc", 12)]), 19)
    assert one.startswith("1 chip left (Bench Boost) and exactly 1 GW (GW19)")


def test_new_window_resets():
    # Everything used in the first half: the second half starts fresh, far from its deadline.
    used = [("wildcard", 6), ("freehit", 10), ("3xc", 12), ("bboost", 15)]
    assert chips.window_warning(_available(19, used), 19) is None
    assert chips.window_warning(_available(20, used), 20) is None
    assert chips.window_warning({}, 19) is None


def _double_and_blank():
    """A pool where clubs 1-6 double in GW7 and blank in GW6; GW8 is a normal week."""
    from tests.test_optimise import make_players
    players = make_players(gws=[6, 7, 8], seed=4)
    moved = players["team"] <= 6
    players.loc[moved, "xp_7"] = players.loc[moved, "xp_7"] * 2
    players.loc[moved, "xp_6"] = 0.0
    players["xp_total"] = players[["xp_6", "xp_7", "xp_8"]].sum(axis=1)
    return players


def test_triple_captain_and_bench_boost_wait_for_the_double():
    from xpfpl.optimise import solve
    players = _double_and_blank()
    plan = solve(players, [6, 7, 8], bank=100.0)
    advice = {a.chip: a for a in chips.advise(players, [6, 7, 8], plan, _available(6), {})}
    assert advice["3xc"].better_later and not advice["3xc"].recommended
    assert advice["bboost"].better_later and not advice["bboost"].recommended
    # The captain in the double week plays twice.
    cap = plan.captains[7]
    assert players.at[cap, "team"] <= 6


def test_a_blank_week_lineup_avoids_blanking_clubs():
    from xpfpl.optimise import solve
    players = _double_and_blank()
    plan = solve(players, [6, 7, 8], bank=100.0)
    xi, bench = players.loc[plan.lineups[6]], players.loc[plan.bench[6]]
    # A blanking starter is only there if nobody of his position on the bench plays this week.
    for pos in (1, 2, 3, 4):
        if (xi[xi["position"] == pos]["xp_6"] == 0).any():
            assert (bench[bench["position"] == pos]["xp_6"] == 0).all()


def _advice(chip, gain, threshold, **kw):
    return chips.ChipAdvice(chip, gain, threshold, kw.pop("expiring", False), "", **kw)


AVAIL = {n: {"stop_event": 19} for n in ("3xc", "bboost", "freehit", "wildcard")}


def test_verdict_close_call_and_no_double_coming():
    v = chips.verdicts([_advice("3xc", 6.1, 6.0)], AVAIL, 11, "Haaland", weeks={})
    assert v[0].action == "close" and v[0].title == "Close call: Triple Captain on Haaland"
    assert "No double gameweeks" in v[0].detail and "after GW19" in v[0].detail
    assert chips.headline(v, 11) == ("close", "GW11: Triple Captain on Haaland, but it's a close call.")


def test_verdict_clear_play_names_the_coming_double():
    weeks = {16: {"doubles": [1, 2], "blanks": []}, 25: {"doubles": [3], "blanks": []}}
    v = chips.verdicts([_advice("3xc", 8.0, 6.0)], AVAIL, 11, "Haaland", weeks=weeks,
                       team_names={1: "ARS", 2: "MCI"})
    assert v[0].action == "play" and "GW16 is a double gameweek (ARS, MCI)" in v[0].detail


def test_verdict_one_chip_a_week_and_reasons_to_save():
    advice = [_advice("bboost", 12.0, 8.0), _advice("3xc", 7.0, 6.0),
              _advice("freehit", 1.5, 18.0),
              _advice("wildcard", 40.0, 30.0, better_later=False)]
    advice.insert(2, chips.ChipAdvice("3xc", 5.0, 6.0, False, "", better_later=True, later_gw=13, later_gain=7.5))
    v = chips.verdicts(advice, AVAIL, 11)
    assert [x.action for x in v] == ["play", "wait", "save", "save", "wait"]
    assert "GW13 looks better (7.5 vs 5.0 now)" in v[2].detail
    assert "Not worth it this week" in v[3].detail
    assert chips.headline(v, 11)[0] == "play"


def test_verdict_expiring_chip_is_played_and_nothing_means_none():
    v = chips.verdicts([_advice("bboost", 3.0, 8.0, expiring=True)], {"bboost": {"stop_event": 11}}, 11)
    assert v[0].action == "last"
    assert chips.headline(chips.verdicts([_advice("freehit", 1.0, 18.0)], AVAIL, 11), 11)[0] == "none"


def test_fixture_weeks_finds_doubles_and_blanks():
    fx = [{"event": 5, "team_h": 1, "team_a": 2}, {"event": 5, "team_h": 3, "team_a": 4},
          {"event": 6, "team_h": 1, "team_a": 2}, {"event": 6, "team_h": 1, "team_a": 3}]
    assert chips.fixture_weeks(fx, 5, 6) == {6: {"doubles": [1], "blanks": [4]}}
