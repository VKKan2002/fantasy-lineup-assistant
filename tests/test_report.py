"""Tests for the swap list: pairing, the toss-up label, and unknown gains staying unknown."""

from ffeval.report import render, swaps

POS = {"a": "WR", "b": "WR", "c": "RB", "d": "TE", "e": "RB"}


def test_same_position_pairs_first():
    pairs = swaps({"a", "c"}, {"b", "e"}, POS, {"a": 5.0, "b": 9.0, "c": 7.0, "e": 7.4})
    assert pairs == [("a", "b", 4.0), ("c", "e", 0.4)]


def test_a_small_gain_is_called_a_toss_up_and_an_unknown_one_is_not_zero():
    pairs = swaps({"a", "c"}, {"b", "e"}, POS, {"a": 5.0, "b": 9.0, "c": 7.0, "e": None})
    text = render(3, pairs, {"b", "e"}, {}, [], [])
    assert "Start b, sit a: +4.0 projected" in text
    assert "Start e, sit c: gain unknown" in text
    small = render(3, [("c", "e", 0.4)], set(), {}, [], [])
    assert "toss-up" in small


def test_no_swaps_says_so():
    assert "Nothing I'd change." in render(3, [], set(), {}, [], [])


def test_the_summary_is_short_and_reads_like_english():
    """Lamb, week 3: twelve facts become three lines. No injury line when there is no
    injury; the spread says who is favoured instead of a signed number."""
    from ffeval.audit.packet import Fact, FactsPacket
    from ffeval.report import summary

    def f(id_, v):
        return Fact(id=id_, label="x", value=v, unit="", source="t", as_of="t", direction="neutral")

    p = FactsPacket(player="CeeDee Lamb", position="WR", team="DAL", opponent="NYG",
                    season=2026, week=3, news=(), facts=(
        f("form.game_w01", 15.4), f("form.game_w02", 35.3), f("form.avg_ppr_l2", 25.35),
        f("injury.report_status", "none"), f("injury.practice_status", "none"),
        f("matchup.spread_line", -3.5), f("matchup.total_line", 52.5),
        f("matchup.is_home", True), f("matchup.roof", None),
        f("defense.wr_ppr_allowed_rank", 14), f("defense.wr_ppr_allowed_per_game", 31.25),
        f("bye.is_bye_week", False)))
    assert summary(p) == [
        "Last games: 15.4, 35.3",
        "vs NYG (home) · underdog by 3.5 · total 52.5",
        "NYG allow the 14th-most points to WRs (31.2 a game)",
    ]
