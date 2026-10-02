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
