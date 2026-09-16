"""ADP-implied expected points.

ADP is a draft rank, not a projection. Every strategy needs points to reason about, so
rank is mapped to points empirically: what does the k-th drafted player at a position
actually score, on average?

    f(position, pos_rank) -> expected PPR total

Fit LEAVE-ONE-SEASON-OUT. The mapping is a fitted model, not a lookup table, and
including the target season would leak that season's outcome into its own projection.

Two smoothing steps, in order:
  1. rolling mean over neighbouring ranks - single ranks are noisy with ~11 seasons
  2. cumulative minimum - enforces that expectation never RISES with a later pick,
     the one thing we know a priori must hold
"""

from __future__ import annotations

import polars as pl

# Games before the prior and the observation carry equal weight. Not tuned - it is the
# value the 91.2% in docs/FINDINGS.md was measured at, so changing it retires that number.
SHRINK_K = 3.0


class ExpectedPoints:
    """f(position, pos_rank) -> expected PPR total, fit excluding one season."""

    def __init__(self, curve: dict[str, list[float]]) -> None:
        # curve[pos][i] = expected points for pos_rank i+1
        self._curve = curve

    def __call__(self, position: str, pos_rank: int) -> float:
        vals = self._curve.get(position)
        if not vals:
            return 0.0
        i = min(max(pos_rank, 1), len(vals)) - 1
        return vals[i]

    @property
    def positions(self) -> list[str]:
        return list(self._curve)

    def replacement(self, position: str, rank: int) -> float:
        """Expected points of the `rank`-th player at a position (the VOR baseline)."""
        return self(position, rank)


def fit_expected_points(
    hist: pl.DataFrame, target_season: int, window: int = 5
) -> ExpectedPoints:
    """hist needs columns: season, position, pos_rank, ppr_total."""
    other = hist.filter(pl.col("season") != target_season)
    agg = (
        other.group_by("position", "pos_rank")
        .agg(pl.col("ppr_total").mean().alias("exp"))
        .sort("position", "pos_rank")
    )
    curve: dict[str, list[float]] = {}
    for pos in agg["position"].unique().to_list():
        c = (
            agg.filter(pl.col("position") == pos)
            .sort("pos_rank")
            .with_columns(
                pl.col("exp")
                .rolling_mean(window_size=window, min_periods=1, center=True)
                .alias("sm")
            )
            .with_columns(pl.col("sm").cum_min().alias("mono"))
        )
        curve[pos] = [float(v) for v in c["mono"].to_list()]
    return ExpectedPoints(curve)


def project_ppg(
    prior_games: list[float], prior_ppg: float | None, k: float = SHRINK_K
) -> float | None:
    """Points per game for one player this week. The measured 91.2% rule.

    Trust the draft market early, trust what the player has actually done as games
    accumulate: the observation gets weight n/(n+k), so at k games they weigh equally.

    Returns None when there is no basis at all - undrafted AND yet to play. That is not
    a zero. A zero projection says "will score nothing", which would bench a waiver
    pickup on the strength of a number nobody computed. The caller has to decide what to
    do with a player the rule cannot rank, and making it decide is the point.

    No matchup term, deliberately. FINDINGS.md section 4 measures PERFECT matchup
    knowledge at 14 points a season over this rule - under a point a week - and a real
    estimate is worth less than perfect. The matchup facts stay in the packet for the
    model to talk about; they do not move the number.
    """
    n = len(prior_games)
    if n == 0:
        return prior_ppg                     # None when undrafted: no basis to project
    observed = sum(prior_games) / n
    if prior_ppg is None:
        return observed                      # undrafted but playing: nothing to shrink to
    w = n / (n + k)
    return w * observed + (1 - w) * prior_ppg
