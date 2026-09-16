"""Build facts packets from nflverse, for a given season and week.

Point-in-time is the whole job. A packet must contain what was knowable before kickoff
and nothing else, so two rules are enforced structurally rather than by care:

  - Only columns in _SCHEDULE_COLS are ever read. load_schedules() carries the final
    score in the same row as the betting line, so a blocklist would hand the model the
    result of the game it is predicting the day nflverse adds a column nobody expected.
    An allowlist fails closed.
  - Only weeks STRICTLY BEFORE the target week feed form and defence facts.

The fact ids and labels match eval/packets/ exactly. Every measured number in
docs/FINDINGS.md is scored against labels tied to those ids, so changing one silently
retires the eval set.

News is derived from the injury table, not the open web: a team-mate ruled out is real,
dated, traceable news, and it is the case docs/DESIGN.md's flowchart cares about - the
starter ahead of someone being Out. Fetching beat writers is the news loop's job and
carries its own failure modes; this needs none of them.
"""

from __future__ import annotations

import nflreadpy as nfl
import polars as pl

from ..audit.packet import Fact, FactsPacket, NewsItem

# Everything a packet may read from a schedule row. The rest of that row - result,
# home_score, away_score, overtime - is the outcome. See the module docstring.
_SCHEDULE_COLS = (
    "game_id", "season", "week", "home_team", "away_team",
    "spread_line", "total_line", "roof",
)

_STATS_SRC = "nflverse player_stats (weekly, REG)"
_SCHED_SRC = "nflverse load_schedules"


def _facts_form(games: list[tuple[int, float]], season: int) -> list[Fact]:
    """The last two games played, plus their average.

    `games` is [(week, ppr)] for weeks before the target, already sorted. A week with no
    row is a week the player did not play - nflverse writes no row for a bye or an
    inactive, while a row of 0.0 means he played and did nothing. That distinction is
    carried through as the string "did not play" rather than dropped: not playing last
    week matters to a start/sit call, and a non-numeric value is skipped by the number
    checkers on its own.
    """
    out: list[Fact] = []
    last_two = games[-2:]
    for week, pts in last_two:
        out.append(Fact(
            id=f"form.game_w{week:02d}",
            label=f"PPR points scored in week {week} (higher is better)",
            value=pts, unit="ppr_points", direction="higher_is_better",
            source=_STATS_SRC, as_of=f"{season} week {week}",
        ))
    played = [p for _, p in last_two if isinstance(p, float)]
    out.append(Fact(
        id="form.avg_ppr_l2",
        label="average PPR points over the two games played so far (higher is better)",
        value=round(sum(played) / len(played), 2) if played else "did not play",
        unit="ppr_points", direction="higher_is_better",
        source=_STATS_SRC, as_of=f"{season} through week {max(w for w, _ in games)}"
        if games else f"{season} no games played",
    ))
    return out


def _facts_injury(row: dict | None, season: int, week: int) -> list[Fact]:
    """Report and practice status. Absent from the report is itself the answer."""
    return [
        Fact(
            id="injury.report_status",
            label="official injury report status: Out, Doubtful, Questionable, "
                  "or none if not listed",
            value=(row or {}).get("report_status") or "none",
            unit="text", direction="neutral",
            source="nflverse load_injuries"
                   + ("" if row else " (not listed on the week %d report)" % week),
            as_of=f"{season} week {week} final report",
        ),
        Fact(
            id="injury.practice_status",
            label="practice participation this week, or none if not listed",
            value=(row or {}).get("practice_status") or "none",
            unit="text", direction="neutral",
            source="nflverse load_injuries"
                   + ("" if row else " (not listed on the week %d report)" % week),
            as_of=f"{season} week {week}",
        ),
    ]


def _facts_matchup(game: dict, team: str, season: int, week: int) -> list[Fact]:
    """Spread, total, venue. Betting lines are set before kickoff, so they are fair game
    for a Sunday lineup call - unlike a preseason draft model, where they would be
    cheating."""
    is_home = game["home_team"] == team
    spread = game["spread_line"] if is_home else -game["spread_line"]
    return [
        Fact(id="matchup.spread_line",
             label="point spread from this team's perspective "
                   "(higher means bigger favourite)",
             value=float(spread), unit="points", direction="higher_is_better",
             source=_SCHED_SRC, as_of=f"{season} week {week} kickoff"),
        Fact(id="matchup.total_line",
             label="over/under total points for the game "
                   "(higher is better for fantasy)",
             value=float(game["total_line"]), unit="points",
             direction="higher_is_better",
             source=_SCHED_SRC, as_of=f"{season} week {week} kickoff"),
        Fact(id="matchup.is_home", label="playing at home", value=is_home,
             unit="yes_no", direction="neutral",
             source=_SCHED_SRC, as_of=f"{season} week {week}"),
        Fact(id="matchup.roof",
             label="stadium roof: outdoors, dome, closed, or open",
             value=game["roof"], unit="text", direction="neutral",
             source=_SCHED_SRC, as_of=f"{season} week {week}"),
    ]


def _facts_defense(
    allowed: dict[str, float], opponent: str, position: str, season: int, week: int
) -> list[Fact]:
    """How generous the opponent has been to this position, through LAST week only.

    Rank 1 = allows the most, so a lower rank is a better matchup. Ranking over whatever
    teams actually appear rather than a hard 32: early in a season, or with a bye, not
    every defence has faced the position yet.
    """
    order = sorted(allowed, key=lambda t: -allowed[t])
    rank = order.index(opponent) + 1 if opponent in order else None
    per_game = allowed.get(opponent)
    src = f"nflverse player_stats, {position} points aggregated by opponent_team"
    asof = f"{season} through week {week - 1}"
    return [
        Fact(id=f"defense.{position.lower()}_ppr_allowed_rank",
             label=f"opponent rank in PPR points allowed to {position}s out of "
                   f"{len(order)}, where 1 = allows the MOST (so a lower rank is a "
                   f"better matchup for our player)",
             value=rank if rank else "no games yet",
             unit=f"rank_of_{len(order)}", direction="lower_is_better",
             source=src, as_of=asof),
        Fact(id=f"defense.{position.lower()}_ppr_allowed_per_game",
             label=f"PPR points the opponent has allowed to opposing {position}s per "
                   f"game (higher is a better matchup for our player)",
             value=round(per_game, 2) if per_game is not None else "no games yet",
             unit="ppr_points", direction="higher_is_better",
             source=src, as_of=asof),
    ]


def _news_teammates(inj: pl.DataFrame, team: str, player_id: str,
                    season: int, week: int) -> list[NewsItem]:
    """Team-mates ruled out or doubtful, as dated prose.

    Structured data rendered into a sentence, so it carries a real date and a real
    source and cannot have been crafted by anyone. The open-web feeds that need
    sanitising are the news loop's problem, not this one's.
    """
    rows = inj.filter(
        (pl.col("team") == team)
        & (pl.col("gsis_id") != player_id)
        & pl.col("report_status").is_in(["Out", "Doubtful"])
    ).to_dicts()
    out = []
    for i, r in enumerate(sorted(rows, key=lambda r: r["full_name"]), start=1):
        # report_primary_injury is a body part ("Ankle"), not a phrase, so it needs the
        # noun - except where nflverse already supplies one ("Illness", "Concussion").
        hurt = (r["report_primary_injury"] or "an undisclosed").lower()
        hurt = hurt if hurt.endswith(("injury", "illness", "concussion")) else f"{hurt} injury"
        out.append(NewsItem(
            id=f"news.injury_{i:02d}",
            text=f"{team} listed {r['position']} {r['full_name']} as "
                 f"{r['report_status'].lower()} for the week {week} game "
                 f"with {'an' if hurt[0] in 'aeiou' else 'a'} {hurt}.",
            url=f"nflverse://load_injuries?season={season}&week={week}&team={team}",
            published=f"{season}-W{week:02d}",
        ))
    return out


def prior_ppg_history(
    player_ids: list[str], season: int, week: int
) -> dict[str, list[float]]:
    """player id -> PPR points in every week before `week`, in order.

    Separate from build_packets on purpose. The projection rule weighs the observation
    by HOW MANY games back it goes, so it needs all of them - while a packet carries only
    the last two, because twelve weeks of individual scores is noise in an email and every
    labelled claim is tied to the twelve facts a packet already has.
    """
    rows = (nfl.load_player_stats(seasons=[season], summary_level="week")
              .filter((pl.col("season_type") == "REG") & (pl.col("week") < week)
                      & pl.col("player_id").is_in(player_ids))
              .sort("week")
              .select("player_id", "fantasy_points_ppr").to_dicts())
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r["player_id"], []).append(float(r["fantasy_points_ppr"] or 0.0))
    return out


def build_packets(player_ids: list[str], season: int, week: int) -> list[FactsPacket]:
    """One packet per player id, for one season and week. Fetches each table once.

    ponytail: no cache. nflverse loads in ~1.6s and nflreadpy caches underneath; a
    cache layer here would be solving a problem that does not exist yet.
    """
    stats = nfl.load_player_stats(seasons=[season], summary_level="week").filter(
        pl.col("season_type") == "REG")
    sched = nfl.load_schedules().select(_SCHEDULE_COLS).filter(
        (pl.col("season") == season) & (pl.col("week") == week))
    inj = nfl.load_injuries(seasons=[season]).filter(pl.col("week") == week)

    prior = stats.filter(pl.col("week") < week)
    who = {r["player_id"]: r for r in stats.filter(
        pl.col("player_id").is_in(player_ids)).sort("week").to_dicts()}

    packets = []
    for pid in player_ids:
        me = who.get(pid)
        if me is None:
            continue                                   # no rows at all this season
        position, name = me["position"], me["player_display_name"]

        # points allowed to this position, by defence, through last week only
        agg = prior.filter(pl.col("position") == position).group_by(
            "opponent_team").agg(
                pl.col("fantasy_points_ppr").sum().alias("pts"),
                pl.col("week").n_unique().alias("g"))
        allowed = {r["opponent_team"]: (r["pts"] or 0.0) / r["g"]
                   for r in agg.to_dicts() if r["g"]}

        mine = prior.filter(pl.col("player_id") == pid).sort("week").to_dicts()
        team = mine[-1]["team"] if mine else me["team"]
        games = [(int(r["week"]), float(r["fantasy_points_ppr"] or 0.0)) for r in mine]

        game = next((g for g in sched.to_dicts()
                     if team in (g["home_team"], g["away_team"])), None)
        on_bye = game is None
        opponent = "" if on_bye else (
            game["away_team"] if game["home_team"] == team else game["home_team"])

        facts = _facts_form(games, season)
        facts += _facts_injury(
            next((r for r in inj.to_dicts() if r["gsis_id"] == pid), None), season, week)
        facts += _facts_matchup(game, team, season, week) if game else []
        facts += _facts_defense(allowed, opponent, position, season, week) if game else []
        facts.append(Fact(
            id="bye.is_bye_week",
            label="on a bye this week and therefore cannot be started",
            value=on_bye, unit="yes_no", direction="neutral",
            source=_SCHED_SRC, as_of=f"{season} week {week}"))

        packets.append(FactsPacket(
            player=name, position=position, team=team, opponent=opponent,
            season=season, week=week, facts=tuple(facts),
            news=tuple(_news_teammates(inj, team, pid, season, week)),
            player_id=pid,
        ))
    return packets
