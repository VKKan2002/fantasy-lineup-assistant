"""Tests for the packet builder. The fetch is not tested; the rules around it are.

nflverse needs a network and takes seconds, so these exercise the parts that decide
what a packet CONTAINS - which is where the point-in-time mistakes live.
"""

from ffeval.ingest.packets import _SCHEDULE_COLS, _facts_defense, _facts_form


def test_the_final_score_is_unreachable():
    """load_schedules() carries the result in the same row as the betting line.

    An allowlist rather than a blocklist, so a column nflverse adds next season is
    excluded until someone decides otherwise instead of leaking by default.
    """
    for outcome in ("result", "home_score", "away_score", "overtime", "away_qb_name"):
        assert outcome not in _SCHEDULE_COLS


def test_a_week_not_played_says_so_rather_than_scoring_zero():
    """No row means a bye or an inactive; a row of 0.0 means he played and did nothing.

    Collapsing those loses the thing a start/sit call most wants to know. The string
    flows through untouched because the number checkers skip non-numeric values.
    """
    facts = {f.id: f.value for f in _facts_form([(1, 38.76)], season=2025)}
    assert facts["form.game_w01"] == 38.76
    assert facts["form.avg_ppr_l2"] == 38.76          # averages only what was played
    assert "form.game_w02" not in facts


def test_the_average_covers_the_last_two_games_only():
    facts = {f.id: f.value for f in _facts_form([(1, 10.0), (2, 20.0), (3, 30.0)], 2025)}
    assert facts["form.avg_ppr_l2"] == 25.0           # weeks 2 and 3, not week 1
    assert "form.game_w01" not in facts


def test_a_defence_that_has_not_faced_the_position_says_so():
    """Early in a season not every defence has played a given position yet. Reporting
    a rank of 0 or a score of 0.0 would read as "allows nothing", the opposite of true."""
    facts = {f.id: f.value for f in _facts_defense({}, "MIA", "QB", 2025, 2)}
    assert facts["defense.qb_ppr_allowed_rank"] == "no games yet"
    assert facts["defense.qb_ppr_allowed_per_game"] == "no games yet"


def test_rank_one_is_the_most_generous_defence():
    allowed = {"MIA": 27.89, "NYJ": 12.0, "NE": 20.0}
    facts = {f.id: f.value for f in _facts_defense(allowed, "MIA", "QB", 2025, 3)}
    assert facts["defense.qb_ppr_allowed_rank"] == 1
    assert facts["defense.qb_ppr_allowed_per_game"] == 27.89
