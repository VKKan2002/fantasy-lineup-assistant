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
