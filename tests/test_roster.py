"""Tests for the ESPN roster read. A recorded reply, trimmed to one team: no managers'
names, no account ids, no cookies."""

import json
from pathlib import Path

from ffeval.ingest.roster import league_rules, my_roster

LEAGUE = json.loads(Path("tests/fixtures/espn_league_trimmed.json").read_text())


def test_the_league_sets_the_slots_not_the_default():
    rules = league_rules(LEAGUE)
    assert rules.starters == {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "D/ST": 1, "K": 1}
    assert rules.flex == 1                         # bench (20) and IR (21) are not slots


def test_roster_reads_slot_position_and_injury():
    roster = {p.name: p for p in my_roster(LEAGUE, team_id=1)}
    assert len(roster) == 15
    assert roster["Josh Jacobs"].slot == "IR"
    assert roster["Zay Flowers"].injury == "QUESTIONABLE"
    assert roster["Browns D/ST"].position == "D/ST" and roster["Browns D/ST"].espn_id < 0


def test_kicker_and_defense_use_the_same_rule_in_league_scoring():
    """19 points over 2 games this season, 140 over 17 last. The defense's game-total fact
    points the other way: a shootout is good for a kicker and bad for a defense."""
    import polars as pl

    from ffeval.ingest.roster import RosterPlayer, kdst_packets

    games = [{"game_id": f"g{s}{w}", "season": s, "week": w, "home_team": "BAL",
              "away_team": "CLE", "spread_line": 3.0, "total_line": 44.5, "roof": "outdoors"}
             for s, w in [(2025, w) for w in range(1, 18)] + [(2026, 1), (2026, 2), (2026, 3)]]
    sched = pl.DataFrame(games)
    k = RosterPlayer("Tyler Loop", 4697745, "K", "K", "ACTIVE", "BAL", {2026: 19.0, 2025: 140.0})
    d = RosterPlayer("Browns D/ST", -16005, "D/ST", "D/ST", "ACTIVE", "CLE", {2026: 4.0})

    packets, history, priors = kdst_packets([k, d], 2026, 3, sched)
    kp, dp = packets
    assert history["espn:4697745"] == [9.5, 9.5]            # 19 / 2 games
    assert priors["espn:4697745"] == round(140 / 17, 2)
    assert "espn:-16005" not in priors                       # no last season: no prior
    assert kp.fact("matchup.total_line").direction == "higher_is_better"
    assert dp.fact("matchup.total_line").direction == "lower_is_better"
    assert dp.opponent == "BAL" and dp.fact("bye.is_bye_week").value is False
    assert kp.fact("injury.report_status").value == "none"
