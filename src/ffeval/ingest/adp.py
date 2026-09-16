"""Draft-position priors for the live projection rule.

project_ppg() needs a number to shrink toward before a player has played: what the draft
market expected of them. That is a rank, not points, so it goes through the same fitted
curve the backtest uses - see models/expected.py.

The curve is fit on the historical pool with the target season held out. For a live season
there are no rows for it yet, so holding it out is free and the fit is honest by
construction rather than by care.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import nflreadpy as nfl
import polars as pl

from ..models.expected import fit_expected_points
from .ids import SKILL_POSITIONS
from .resolve import build_crosswalk, resolve, season_activity

URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr"
POOL = Path("checks/out/pool_full.parquet")


def adp_priors(season: int, pool_path: Path = POOL, teams: int = 12) -> dict[str, float]:
    """player id -> expected PPR points per game, implied by this season's draft position.

    A player absent from the returned mapping has no draft position - undrafted, or a
    name that did not resolve. project_ppg takes None for those and says so rather than
    guessing; see the rule in docs/DESIGN.md about lookup failure and real absence being
    different things.

    ponytail: no cache. One HTTP call and one parquet read per run, on a weekly job.
    """
    if not pool_path.exists():
        raise FileNotFoundError(
            f"{pool_path} is missing - build it first:\n"
            f"    uv run python checks/step1_build_pool.py"
        )
    hist = pl.read_parquet(pool_path)
    curve = fit_expected_points(
        hist.select("season", "position", "pos_rank",
                    pl.col("ppg").alias("ppr_total")),
        target_season=season,
    )

    r = httpx.get(URL, params={"teams": teams, "year": season, "position": "all"},
                  timeout=30)
    r.raise_for_status()
    drafted = (pl.DataFrame(r.json()["players"])
                 .filter(pl.col("position").is_in(SKILL_POSITIONS))
                 .sort("adp")
                 .with_columns(season=pl.lit(season)))

    stats = nfl.load_player_stats(seasons=[season], summary_level="week").filter(
        pl.col("season_type") == "REG")
    matched = resolve(drafted, build_crosswalk(nfl.load_ff_playerids()),
                      season_activity(stats), season)

    ranked = (matched.filter(pl.col("gsis_id").is_not_null())
                     .with_columns(pos_rank=pl.col("adp").rank("ordinal")
                                              .over("position").cast(pl.Int32)))
    return {row["gsis_id"]: curve(row["position"], int(row["pos_rank"]))
            for row in ranked.to_dicts()}
