"""Tests for the loop's building blocks and the lineup decision. No model anywhere.

The loop itself is tested in test_graph.py. PACKET, _section and _verdicts live here
because both files use them.
"""

from pathlib import Path

from ffeval.audit.packet import Fact, FactsPacket
from ffeval.audit.verdicts import AuditResult, ClaimVerdict, Verdict
from ffeval.pipeline import _apply, decide_starters
from ffeval.writer import PlayerSection

PACKET = FactsPacket.load(Path("eval/packets/2025_w03_allen.json"))


def _section(*prose):
    return PlayerSection(player="Josh Allen", decision="start",
                         templated=("a: 1",), prose=prose, dropped=())


def _verdicts(*pairs):
    return AuditResult(
        verdicts=tuple(ClaimVerdict(claim=c, verdict=v, evidence_ids=(), reason="why")
                       for c, v in pairs),
        model="fake", prompt_version=1)


def test_apply_swaps_and_removes():
    s = _section("keep me", "replace me", "delete me")
    out = _apply(s, {"replace me": "replaced", "delete me": ""})
    assert out.prose == ("keep me", "replaced")
    assert out.templated == s.templated          # numbers are untouched by the loop


# ------------------------------------------------- choosing the lineup (no model at all)

def _pkt(name, position, on_bye=False, pid=None):
    return FactsPacket(
        player=name, position=position, team="BUF", opponent="MIA", season=2025, week=3,
        facts=(Fact(id="bye.is_bye_week", label="on a bye", value=on_bye, unit="yes_no",
                    source="test", as_of="2025 week 3", direction="neutral"),),
        news=(), player_id=pid or name.lower(),
    )


ROSTER = [_pkt("qb1", "QB"), _pkt("rb1", "RB"), _pkt("rb2", "RB"), _pkt("rb3", "RB"),
          _pkt("wr1", "WR"), _pkt("wr2", "WR"), _pkt("te1", "TE")]


def test_the_best_projected_players_start():
    """Eight players, seven slots, so exactly one sits - and it is the worst projection.

    rb1 and rb3 take the two RB slots, wr1 and wr2 the WR slots, and the FLEX goes to
    the best of what is left. rb2 is the only one with nowhere to go. No model involved
    anywhere in this.
    """
    roster = ROSTER + [_pkt("wr9", "WR")]
    proj = {"qb1": 20.0, "rb1": 18.0, "rb2": 5.0, "rb3": 15.0,
            "wr1": 14.0, "wr2": 13.0, "te1": 9.0, "wr9": 12.0}
    starters, _ = decide_starters(roster, proj, {})

    assert len(starters) == 7                       # 1QB 2RB 2WR 1TE 1FLEX
    assert "wr9" in starters                        # takes the FLEX
    assert "rb2" not in starters                    # the only one benched


def test_a_player_on_a_bye_cannot_start():
    """best_lineup only sees a number, so the bye has to be filtered before it."""
    roster = ROSTER + [_pkt("wr3", "WR", on_bye=True)]
    proj = {p.player_id: 10.0 for p in roster} | {"wr3": 99.0}
    starters, _ = decide_starters(roster, proj, {})
    assert "wr3" not in starters


def test_an_unprojectable_player_is_skipped_and_reported():
    """Undrafted and yet to play. Zero would be a claim; None is the truth, and it comes
    back so the caller can say who was left out."""
    proj = {p.player_id: 10.0 for p in ROSTER}
    roster = ROSTER + [_pkt("rookie", "WR")]        # no prior, no history
    starters, projections = decide_starters(roster, proj, {})
    assert projections["rookie"] is None
    assert "rookie" not in starters


def test_a_player_ruled_out_cannot_start():
    """He has a fine projection and physically cannot play.

    best_lineup() only sees a number, so Out has to be filtered before it runs - this is
    the bug that put CeeDee Lamb in a starting lineup next to a note saying he was out.
    """
    out = FactsPacket(
        player="lamb", position="WR", team="DAL", opponent="NYG", season=2025, week=6,
        facts=(Fact(id="injury.report_status", label="status", value="Out", unit="text",
                    source="test", as_of="2025 week 6", direction="neutral"),),
        news=(), player_id="lamb")
    roster = ROSTER + [out]
    proj = {p.player_id: 10.0 for p in roster} | {"lamb": 99.0}

    starters, projections = decide_starters(roster, proj, {})
    assert "lamb" not in starters
    assert projections["lamb"] == 99.0        # projected fine; simply not eligible


def test_questionable_does_not_block_a_start():
    """Doubtful and Questionable are probabilities. The projection handles uncertainty;
    the model comments on it. Neither overrules the lineup."""
    q = FactsPacket(
        player="q", position="WR", team="DAL", opponent="NYG", season=2025, week=6,
        facts=(Fact(id="injury.report_status", label="status", value="Questionable",
                    unit="text", source="test", as_of="2025 week 6", direction="neutral"),),
        news=(), player_id="q")
    starters, _ = decide_starters([q], {"q": 20.0}, {})
    assert "q" in starters


def test_locked_players_stay_where_they_are():
    """Thursday's game is over by Sunday. A locked starter stays in however badly he is
    projected, and a locked bench player cannot come in however well."""
    proj = {p.player_id: 10.0 for p in ROSTER} | {"rb2": 1.0}
    roster = ROSTER + [_pkt("wr9", "WR")]
    proj["wr9"] = 99.0
    starters, projections = decide_starters(roster, proj, {},
                                            must_start={"rb2"}, cannot_start={"wr9"})
    assert "rb2" in starters and "wr9" not in starters
    assert projections["rb2"] == 1.0                 # the real number, not the lock trick
