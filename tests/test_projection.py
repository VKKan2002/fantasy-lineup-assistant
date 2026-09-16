"""Tests for the weekly projection rule. Pure arithmetic, no data, no network."""

from ffeval.models.expected import SHRINK_K, project_ppg


def test_week_one_is_the_draft_market():
    """No games played yet, so there is nothing to shrink toward the market FROM."""
    assert project_ppg([], 18.0) == 18.0


def test_no_draft_position_and_no_games_cannot_be_projected():
    """None, not 0.0. A zero says "will score nothing" and would bench a waiver pickup
    on a number nobody computed; None makes the caller decide."""
    assert project_ppg([], None) is None


def test_an_undrafted_player_who_has_played_is_judged_on_what_he_did():
    """Nothing to shrink toward, so the observation stands on its own."""
    assert project_ppg([10.0, 20.0], None) == 15.0


def test_at_k_games_the_market_and_the_record_weigh_the_same():
    """w = n/(n+k), so n == k is the crossover. With k=3: half observed, half prior."""
    assert project_ppg([20.0] * int(SHRINK_K), 10.0) == 15.0


def test_the_record_takes_over_as_games_accumulate():
    """Same player, same prior, more evidence - the projection moves toward observed."""
    prior, observed = 10.0, 20.0
    early = project_ppg([observed], prior)
    late = project_ppg([observed] * 9, prior)
    assert early < late < observed


def test_the_weighting_is_exactly_n_over_n_plus_k():
    n = 9
    got = project_ppg([20.0] * n, 10.0)
    w = n / (n + SHRINK_K)
    assert got == w * 20.0 + (1 - w) * 10.0
