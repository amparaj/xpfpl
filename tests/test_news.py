import pandas as pd
import pytest

from xpfpl import config
from xpfpl.data import news

PAGE = """
<html><body><p class="last-updated">Last updated on October 8th, 2026</p>
<div><h2><img src="x.png"><noscript><img src="y.png"></noscript> Arsenal</h2>
<p><span style="font-weight: 600;">OUT</span><br>
William Saliba</p>
<p><span style="font-weight: 600;">DOUBT</span><br>
Bukayo Saka</p>
<p><span style="font-weight: 600;">IN</span><br>
Ben White<br>
Nobody Known</p>
<blockquote>&#8220;They are all good.&#8221; &#8211; Mikel Arteta</blockquote>
</div>
<div><h2> Manchester City</h2>
<p><span style="font-weight: 600;">OUT</span><br>
Jos&eacute; Ruben Dias</p>
</div>
<footer><h2>Not a club</h2></footer>
</body></html>
"""


def element(id_, first, second, web, team, status="a", chance=None, news_="", added=None, link=None, risks=None):
    return {"id": id_, "code": 1000 + id_, "first_name": first, "second_name": second, "web_name": web, "team": team,
            "element_type": 2, "status": status, "chance_of_playing_next_round": chance,
            "chance_of_playing_this_round": chance, "news": news_, "news_added": added, "scout_news_link": link,
            "scout_risks": risks or [], "selected_by_percent": "1.0", "now_cost": 50}


@pytest.fixture
def bs():
    return {
        "teams": [{"id": 1, "code": 3, "name": "Arsenal", "short_name": "ARS"},
                  {"id": 2, "code": 43, "name": "Man City", "short_name": "MCI"}],
        "events": [{"id": 6, "deadline_time": "2026-10-03T10:00:00Z", "is_current": True, "is_next": False},
                   {"id": 7, "deadline_time": "2026-10-10T10:00:00Z", "is_current": False, "is_next": True},
                   {"id": 8, "deadline_time": "2026-10-17T10:00:00Z", "is_current": False, "is_next": False}],
        "elements": [
            element(1, "William", "Saliba", "Saliba", 1, "i", 0, "Back injury - Expected back 17 Oct",
                    "2026-10-05T10:00:00Z", "https://club.example/news?id=4&utm_source=pl"),
            element(2, "Bukayo", "Saka", "Saka", 1),
            element(3, "Benjamin", "White", "White", 1, "d", 25, "Knock - 25% chance of playing", "2026-10-06T10:00:00Z"),
            element(4, "Rúben", "Gato Alves Dias", "Rúben", 2, "s", 0, "Suspended until 17 Oct", "2026-10-09T10:00:00Z"),
            element(5, "Loan", "Player", "Loanee", 2, risks=[{"property": "loan_ineligible", "notes": "Parent club",
                                                              "gameweek": 8, "url": None}]),
        ],
    }


def test_parse_press_reads_clubs_statuses_and_quotes():
    press = news.parse_press(PAGE)
    assert press["updated"] == "2026-10-08"
    assert [c["club"] for c in press["clubs"]] == ["Arsenal", "Manchester City"]
    arsenal = press["clubs"][0]
    assert [(p["name"], p["status"]) for p in arsenal["players"]] == [
        ("William Saliba", "OUT"), ("Bukayo Saka", "DOUBT"), ("Ben White", "IN"), ("Nobody Known", "IN")]
    assert arsenal["quotes"] == [{"text": "They are all good.", "by": "Mikel Arteta"}]


def test_match_press_links_players_by_club_and_name(bs):
    press = news.match_press(news.parse_press(PAGE), bs)
    arsenal, city = press["clubs"]
    assert arsenal["team"] == 1 and city["team"] == 2 and city["team_code"] == 43
    ids = {p["name"]: p["id"] for p in arsenal["players"]}
    assert ids == {"William Saliba": 1, "Bukayo Saka": 2, "Ben White": 3, "Nobody Known": None}
    assert city["players"][0]["id"] == 4          # accents and a longer FPL name


def test_return_date_and_kind():
    assert news.return_date("Hamstring injury - Expected back 10 Oct", "2026-09-20T10:00:00Z") == "2026-10-10"
    assert news.return_date("Suspended until 3 Jan", "2026-12-20T10:00:00Z") == "2027-01-03"
    assert news.return_date("Knee injury - Unknown return date", "2026-09-20T10:00:00Z") is None
    assert news.kind("s", "Suspended until 17 Oct") == "suspension"
    assert news.kind("u", "Has joined Juventus on loan for the rest of the season") == "left"
    assert news.kind("i", "Knee injury - Unknown return date") == "injury"
    assert news.kind("d", "Knock - 75% chance of playing") == "doubt"
    assert news.reason("Knee injury - Unknown return date") == "Knee injury"
    assert news.reason("Suspended until 17 Oct") == "Suspended"


def test_fpl_table_cleans_links_and_keeps_risks(bs):
    t = news.fpl(bs).set_index("id")
    assert list(t.index) == [1, 3, 4, 5]                        # Saka has no news
    assert t.loc[1, "source"] == "https://club.example/news?id=4"
    assert t.loc[1, "back"] == "2026-10-17" and t.loc[4, "kind"] == "suspension"
    assert t.loc[5, "risks"][0]["gw"] == 8


def _frame(bs):
    rows = []
    kickoffs = {7: "2026-10-11T14:00:00Z", 8: "2026-10-18T14:00:00Z"}
    for e in bs["elements"]:
        for gw, ko in kickoffs.items():
            rows.append({"element": e["id"], "gw": gw, "status": e["status"],
                         "chance_of_playing_next_round": e["chance_of_playing_next_round"], "kickoff_time": ko})
    return pd.DataFrame(rows)


def _avail(bs, press=None):
    f = _frame(bs)
    a = f.join(news.availability(f, bs, 7, press))
    return {(r.element, r.gw): (round(r.avail, 3), r.news_rule) for r in a.itertuples()}


def test_availability_without_press(bs):
    a = _avail(bs)
    assert a[(1, 7)] == (0.0, "flag")                           # next week: FPL's flag stands
    assert a[(1, 8)] == (config.RETURN_CHANCE, "back")          # back on the 17th, match on the 18th
    assert a[(3, 7)] == (0.25, "flag") and a[(3, 8)] == (0.5, "flag")
    assert a[(4, 8)] == (1.0, "back")                           # ban over
    assert a[(5, 7)] == (1.0, "flag") and a[(5, 8)] == (0.0, "risk")


def test_return_date_after_the_match_keeps_him_out(bs):
    bs["elements"][0]["news"] = "Back injury - Expected back 25 Oct"
    assert _avail(bs)[(1, 8)] == (0.0, "back")


def test_fresh_press_overrides_older_fpl_news(bs):
    press = news.match_press(news.parse_press(PAGE), bs)        # 8 Oct, after the GW6 deadline
    a = _avail(bs, press)
    assert a[(2, 7)] == (config.PRESS_DOUBT_CHANCE, "press")    # Saka: no FPL news, press DOUBT
    assert a[(3, 7)] == (1.0, "press")                          # White: FPL 25% on the 6th, press IN on the 8th
    assert a[(1, 7)] == (0.0, "press")                          # Saliba: OUT either way
    assert a[(4, 7)] == (0.0, "flag")                           # Dias: FPL updated the 9th, after the presser


def test_stale_press_is_ignored(bs):
    press = news.match_press(news.parse_press(PAGE.replace("October 8th", "October 1st")), bs)
    assert not news.press_is_for_next(press, bs)
    assert _avail(bs, press)[(3, 7)] == (0.25, "flag")


def test_press_switch_off(bs, monkeypatch):
    monkeypatch.setattr(config, "PRESS_NEWS", False)
    monkeypatch.setattr(config, "NEWS_RETURN_DATES", False)
    press = news.match_press(news.parse_press(PAGE), bs)
    a = _avail(bs, press)
    assert a[(3, 7)] == (0.25, "flag") and a[(1, 8)] == (0.25, "flag") and a[(5, 8)] == (1.0, "flag")


def test_update_logs_changes_and_recoveries(bs):
    t0 = pd.Timestamp("2026-10-07T12:00:00Z")
    press = news.match_press(news.parse_press(PAGE), bs)
    first = news.update(None, bs, press=press, now=t0)
    assert {c["id"] for c in first["log"]} == {1, 3, 4}         # everyone with news; the loanee has none
    assert first["log"][0]["t"] == "2026-10-05T10:00:00Z"        # FPL's own time
    assert first["press"]["fresh"] and {p["id"]: p["press"] for p in first["players"]}[1] == "OUT"

    same = news.update(first, bs, now=t0 + pd.Timedelta(hours=1), read_press=False)
    assert len(same["log"]) == 3 and same["press"]["updated"] == "2026-10-08"

    bs["elements"][2].update(status="a", chance=None, news="", news_added="2026-10-08T15:00:00Z")
    bs["elements"][2]["chance_of_playing_next_round"] = None
    bs["elements"][0].update(chance_of_playing_next_round=75, news="Back injury - 75% chance of playing",
                             news_added="2026-10-08T16:00:00Z")
    later = news.update(same, bs, now=t0 + pd.Timedelta(hours=2), read_press=False)
    new = later["log"][3:]
    assert {(c["id"], c["news"]) for c in new} == {(1, "Back injury - 75% chance of playing"), (3, "")}
    assert 3 not in {p["id"] for p in later["players"]}


def test_new_season_starts_a_new_log(bs):
    first = news.update(None, bs, press={"updated": None, "clubs": []}, now=pd.Timestamp("2026-10-07T12:00:00Z"))
    first["season"] = "2025-26"
    again = news.update(first, bs, press={"updated": None, "clubs": []}, now=pd.Timestamp("2026-10-07T13:00:00Z"))
    assert len(again["log"]) == 3


def test_scorecard_scores_news_rows():
    from xpfpl import scorecard
    pred = pd.DataFrame({"chance": [0, 50, None, None], "press": ["OUT", None, "IN", None],
                         "avail": [0.0, 0.5, 1.0, 1.0]}, index=pd.Index([1, 2, 3, 4], name="element"))
    actual = pd.DataFrame({"total_points": [0, 2, 6, 1], "minutes": [0, 90, 90, 0]}, index=pred.index)
    rows = {r["kind"]: r for r in scorecard._news_rows(pred, actual, 7, "ensemble")}
    assert rows["press OUT"]["played"] == 0 and rows["press IN"]["played"] == 1
    assert rows["FPL flag"]["n"] == 1 and rows["FPL flag"]["avail"] == 0.5
    assert "press DOUBT" not in rows
