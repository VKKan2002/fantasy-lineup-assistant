"""This week's report for your real team: `uv run python -m ffeval [--email]`.

Without --email it prints the report. With it, it sends the report and prints only the
message id: the scheduled run's log can be read by anyone who can see the repo, and the
report is your roster.

ESPN roster -> facts packets -> news and the digging agent -> lineup -> write, audit,
rewrite -> plain text. Every piece lives elsewhere; this file only connects them, in
that order, so the order is readable in one place.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import nflreadpy as nfl
import polars as pl

from .graph import GRAPH
from .ingest.news import add_news
from .ingest.packets import _SCHEDULE_COLS, build_packets, prior_ppg_history
from .ingest.roster import fetch_league, kdst_packets, league_rules, my_roster
from .pipeline import _can_play, decide_starters
from .mail import html_report, send
from .report import render, swaps

SKILL = ("QB", "RB", "WR", "TE")


def kicked_off(season: int, week: int, now: datetime) -> set[str]:
    """Teams whose game this week has started. nflverse gives kickoff in US Eastern."""
    games = nfl.load_schedules().select(_SCHEDULE_COLS).filter(
        (pl.col("season") == season) & (pl.col("week") == week)).to_dicts()
    out = set()
    for g in games:
        kick = datetime.fromisoformat(f"{g['gameday']}T{g['gametime']}").replace(
            tzinfo=ZoneInfo("America/New_York"))
        if kick <= now:
            out |= {g["home_team"], g["away_team"]}
    return out


def main() -> None:
    # ponytail: calendar year. January playoff weeks belong to the previous season; fix
    # when the report is ever run in January.
    season = datetime.now().year
    league = fetch_league(season)
    week = league["scoringPeriodId"]
    roster = my_roster(league)
    notes = [f"On IR, not considered: {r.name}" for r in roster if r.slot == "IR"]
    roster = [r for r in roster if r.slot != "IR"]      # the league will not start them

    # ESPN id -> nflverse id, by id, never by name.
    skill = [r for r in roster if r.position in SKILL]
    xw = nfl.load_ff_playerids().filter(
        pl.col("espn_id").is_in([r.espn_id for r in skill]) & pl.col("gsis_id").is_not_null())
    gsis = {int(r["espn_id"]): r["gsis_id"] for r in xw.select("espn_id", "gsis_id").to_dicts()}

    packets = build_packets([gsis[r.espn_id] for r in skill if r.espn_id in gsis], season, week)
    have = {p.player_id for p in packets}
    notes += [f"No nflverse data yet, not considered: {r.name}" for r in skill
              if gsis.get(r.espn_id) not in have]

    kd, kd_history, kd_priors = kdst_packets(roster, season, week)
    history = prior_ppg_history(list(have), season, week) | kd_history
    # Fork 18c: ESPN's projected season total, per game, as the prior for QB/RB/WR/TE.
    priors = {gsis[r.espn_id]: r.projection / 17 for r in skill
              if r.espn_id in gsis and r.projection} | kd_priors

    packets = add_news(packets + kd)

    # Who the manager has in the lineup today, in the packets' names (nflverse spells
    # "Deebo Samuel", ESPN "Deebo Samuel Sr." - joined by id, so the name follows).
    name = {p.player_id: p.player for p in packets}
    current = {name.get(gsis.get(r.espn_id), name.get(f"espn:{r.espn_id}"))
               for r in roster if r.slot != "BENCH"} - {None}
    # Thursday's players are locked by Sunday: keep them where ESPN has them.
    started = kicked_off(season, week, datetime.now(timezone.utc))
    locked = {p.player for p in packets if p.team in started}
    starters, projections = decide_starters(
        packets, priors, history, league_rules(league),
        must_start=locked & current, cannot_start=locked - current)
    if locked:
        notes.append(f"Game already started, left where they are: {', '.join(sorted(locked))}")
    value = {p.player: (projections[p.player] if _can_play(p) else 0.0) for p in packets}
    pairs = swaps(current, starters, {p.player: p.position for p in packets}, value)

    out = GRAPH.invoke({"packets": packets, "starters": starters})
    fallback = out.get("fallback_reason", "")
    by_name = {p.player: p for p in packets}
    text = render(week, pairs, starters, projections, out["sections"], notes, fallback,
                  by_name)
    if "--email" not in sys.argv:
        print(text)
        return
    # The badge shows ESPN's status, which is live midweek; nflverse's official one only
    # appears on Friday. Display only - the lineup rule still reads the packet.
    status = {name.get(gsis.get(r.espn_id), r.name): r.injury.replace("_", " ").title()
              for r in roster}
    html = html_report(week, pairs, starters, projections, out["sections"], notes,
                       status, fallback, by_name)
    # The date keeps every report in its own thread. With a repeated subject Gmail threads
    # them and folds text the earlier email already had - the swap box went missing that way.
    day = datetime.now(ZoneInfo("America/New_York")).strftime("%a %b %-d")
    subject = (f"Week {week} ({day}): {len(pairs)} change{'s' * (len(pairs) != 1)} to consider"
               if pairs else f"Week {week} ({day}): nothing I'd change")
    print("sent", send(subject, html, text))


if __name__ == "__main__":
    main()
