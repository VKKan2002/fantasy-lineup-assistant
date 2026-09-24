"""Your real team and your league's rules, from ESPN. One request.

The league is private, so the request carries two cookies from a logged-in browser
(ESPN_S2, ESPN_SWID in the gitignored .env). They are a password to the ESPN account:
they never go in a prompt, a log line, or a commit.

Players are joined to nflverse by ESPN id through nflverse's own id table, never by name.
Joining by name is how a Hall of Fame back's season got credited to his son, 81 times.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

import nflreadpy as nfl
import polars as pl

from ..audit.packet import Fact, FactsPacket
from ..scoring.league import League
from .news import ESPN_TEAM, _get
from .packets import _SCHED_SRC, _SCHEDULE_COLS, _facts_matchup

NFLVERSE_TEAM = {v: k for k, v in ESPN_TEAM.items()}   # ESPN proTeamId -> "BAL"

LEAGUE_URL = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
              "/segments/0/leagues/{league}?view=mRoster&view=mSettings&view=mTeam")

# ESPN's numeric codes. Fixed by ESPN, so tables, not lookups.
POSITION = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST"}
SLOT = {0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "D/ST", 17: "K", 20: "BENCH", 21: "IR",
        23: "FLEX"}


@dataclass(frozen=True)
class RosterPlayer:
    name: str
    espn_id: int          # negative for a team defense: -16005 is the Browns D/ST
    position: str
    slot: str             # where the manager has him right now
    injury: str           # ESPN's status: ACTIVE, QUESTIONABLE, OUT, ...
    team: str = ""        # nflverse abbreviation
    # season -> actual fantasy points, already in THIS league's scoring (ESPN applies it)
    points: dict[int, float] = field(default_factory=dict)
    # ESPN's projected season total, league scoring. The prior for QB/RB/WR/TE (Fork 18c):
    # there is no preseason ADP for 2026 - the ADP site only serves in-season drafts,
    # which already count weeks 1-2 and would count them twice.
    projection: float | None = None


def fetch_league(season: int) -> dict:
    """The whole league as ESPN sends it. Needs ESPN_S2, ESPN_SWID, ESPN_LEAGUE_ID."""
    from dotenv import load_dotenv

    load_dotenv()
    return _get(LEAGUE_URL.format(season=season, league=os.environ["ESPN_LEAGUE_ID"]),
                headers={"Cookie": f"espn_s2={os.environ['ESPN_S2']}; "
                                   f"SWID={os.environ['ESPN_SWID']}"})


def my_roster(league: dict, team_id: int | None = None) -> list[RosterPlayer]:
    """The manager's players, with the slot each is in today."""
    team_id = team_id or int(os.environ["ESPN_TEAM_ID"])
    team = next(t for t in league["teams"] if t["id"] == team_id)
    season = league.get("seasonId")
    out = []
    for e in team["roster"]["entries"]:
        pl = e["playerPoolEntry"]["player"]
        out.append(RosterPlayer(
            name=pl["fullName"], espn_id=int(pl["id"]),
            position=POSITION.get(pl["defaultPositionId"], "?"),
            slot=SLOT.get(e["lineupSlotId"], "?"),
            injury=pl.get("injuryStatus") or "ACTIVE",
            team=NFLVERSE_TEAM.get(pl.get("proTeamId"), ""),
            points={s["seasonId"]: float(s["appliedTotal"]) for s in pl.get("stats", [])
                    if s["statSourceId"] == 0 and s["statSplitTypeId"] == 0},
            projection=next((float(s["appliedTotal"]) for s in pl.get("stats", [])
                             if s["seasonId"] == season and s["statSourceId"] == 1
                             and s["statSplitTypeId"] == 0), None)))
    return out


def league_rules(league: dict) -> League:
    """Starting slots from the league's own settings (Fork 14a). best_lineup() is only
    optimal for the slots it is given; the wrong slots make it the best answer to the
    wrong question."""
    counts = {int(k): v for k, v in
              league["settings"]["rosterSettings"]["lineupSlotCounts"].items() if v}
    starters = {SLOT[k]: n for k, n in counts.items() if SLOT.get(k) in
                ("QB", "RB", "WR", "TE", "K", "D/ST")}
    return League(starters=starters, flex=counts.get(23, 0))


def kdst_packets(
    roster: list[RosterPlayer], season: int, week: int, sched: pl.DataFrame | None = None,
) -> tuple[list[FactsPacket], dict[str, list[float]], dict[str, float]]:
    """Kickers and defenses: packets, plus the history and prior decide_starters() wants.

    Same rule as everyone (Fork 16a): this season's points per game, shrunk toward last
    season's. The points are ESPN's, in this league's own scoring - which matters most for
    a defense, whose points depend on league-specific rules for sacks and points allowed.
    The rule was measured on QB/RB/WR/TE only; nothing here says it is right for these.

    Short packets (Fork 17a): points per game, the matchup, the bye. The game total is the
    fact that matters for both, in opposite directions.
    """
    if sched is None:
        sched = nfl.load_schedules().select(_SCHEDULE_COLS)
    sched = sched.filter(pl.col("season").is_in([season - 1, season]) & (pl.col("week") <= 18))
    packets, history, priors = [], {}, {}
    for r in roster:
        if r.position not in ("K", "D/ST"):
            continue
        pid = f"espn:{r.espn_id}"          # no nflverse id: a defense is a team, not a player
        mine = sched.filter((pl.col("home_team") == r.team) | (pl.col("away_team") == r.team))
        # ponytail: games = team games, so a kicker who missed one is averaged low. Weekly
        # ESPN points would fix it, at one request per week.
        n_now = mine.filter((pl.col("season") == season) & (pl.col("week") < week)).height
        n_last = mine.filter(pl.col("season") == season - 1).height

        facts = []
        for yr, n, key, when in ((season, n_now, "this", "this season"),
                                 (season - 1, n_last, "last", "last season")):
            if n and yr in r.points:
                ppg = round(r.points[yr] / n, 2)
                facts.append(Fact(
                    id=f"espn.points_per_game_{key}_season",
                    label=f"fantasy points per game {when}, in this league's scoring",
                    value=ppg, unit="points", direction="higher_is_better",
                    source="ESPN league scoring", as_of=f"{season} week {week}"))
                if key == "this":
                    history[pid] = [ppg] * n   # project_ppg reads only the mean and count
                else:
                    priors[pid] = ppg

        game = next(iter(mine.filter((pl.col("season") == season)
                                     & (pl.col("week") == week)).to_dicts()), None)
        if game:
            matchup = _facts_matchup(game, r.team, season, week)
            if r.position == "D/ST":                 # a high-scoring game hurts a defense
                matchup = [replace(f, label="over/under total points for the game "
                                            "(lower is better for a defense)",
                                   direction="lower_is_better")
                           if f.id == "matchup.total_line" else f for f in matchup]
            facts += matchup
        facts.append(Fact(
            id="bye.is_bye_week", label="on a bye this week and therefore cannot be started",
            value=game is None, unit="yes_no", direction="neutral",
            source=_SCHED_SRC, as_of=f"{season} week {week}"))
        if r.position == "K":
            status = {"OUT": "Out", "DOUBTFUL": "Doubtful",
                      "QUESTIONABLE": "Questionable"}.get(r.injury, "none")
            facts.append(Fact(
                id="injury.report_status", label="official injury report status",
                value=status, unit="text", direction="neutral",
                source="ESPN injury status", as_of=f"{season} week {week}"))

        opponent = "" if not game else (
            game["away_team"] if game["home_team"] == r.team else game["home_team"])
        packets.append(FactsPacket(
            player=r.name, position=r.position, team=r.team, opponent=opponent,
            season=season, week=week, facts=tuple(facts), news=(), player_id=pid))
    return packets, history, priors
