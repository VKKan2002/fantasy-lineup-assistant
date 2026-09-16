"""Tests for the rewrite loop. The model calls are faked; the loop's decisions are not.

What matters here is termination and what survives: that a clean pass costs nothing, that
a sentence which never comes clean is cut rather than shipped, and that the budget is
actually a bound.
"""

from pathlib import Path

import ffeval.pipeline as pipeline
from ffeval.audit.auditor import QuotaExhausted
from ffeval.audit.packet import Fact, FactsPacket
from ffeval.audit.verdicts import AuditResult, ClaimVerdict, Verdict
from ffeval.pipeline import BUDGET, _apply, decide_starters, run
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


def _fake(monkeypatch, sections, audits, replacements):
    """Wire up fake write/audit/rewrite and count the calls each one gets."""
    calls = {"audit": 0, "rewrite": 0}

    def fake_audit(items, model):
        """One call for the whole roster now, so one result per packet comes back."""
        calls["audit"] += 1
        return [audits.pop(0) for _ in items]

    def fake_rewrite(items, model=None):
        calls["rewrite"] += 1
        return [replacements.pop(0) for _ in items]

    monkeypatch.setattr(pipeline, "write", lambda p, s, m: list(sections))
    monkeypatch.setattr(pipeline, "audit_roster", fake_audit)
    monkeypatch.setattr(pipeline, "rewrite", fake_rewrite)
    return calls


def test_a_clean_pass_never_calls_the_writer_back(monkeypatch):
    """Nothing flagged means no rewrite call at all - the loop costs one audit."""
    calls = _fake(monkeypatch,
                  [_section("He is a strong play.")],
                  [_verdicts(("He is a strong play.", Verdict.SUPPORTED))],
                  [])
    out = run([PACKET], {"Josh Allen"})
    assert calls["rewrite"] == 0
    assert out.sections[0].prose == ("He is a strong play.",)
    assert out.rewrites == ()


def test_a_sentence_that_never_comes_clean_is_cut_not_shipped(monkeypatch):
    """Rejected every time. After BUDGET rounds it is deleted, and the log says so.

    Shipping an unsupported claim is the one outcome this loop exists to prevent, so a
    sentence that outlasts its budget must not survive it.
    """
    bad = "He is a focal point of the offense."
    calls = _fake(
        monkeypatch,
        [_section(bad)],
        [_verdicts((bad, Verdict.NOT_IN_PACKET))] * (BUDGET + 1),
        [bad] * BUDGET,                       # writer keeps handing back the same thing
    )
    out = run([PACKET], {"Josh Allen"})

    assert out.sections[0].prose == ()                    # cut
    assert calls["rewrite"] == BUDGET                     # budget is a real bound
    assert out.rewrites[-1].outcome == "deleted"
    assert [r.outcome for r in out.rewrites] == ["replaced"] * BUDGET + ["deleted"]


def test_an_empty_replacement_drops_the_sentence(monkeypatch):
    """The writer is allowed to say there is nothing supportable to write. That beats
    an invented replacement, so "" removes the sentence instead of blanking it."""
    bad = "He is a focal point of the offense."
    _fake(monkeypatch, [_section(bad, "Good matchup.")],
          [_verdicts((bad, Verdict.NOT_IN_PACKET), ("Good matchup.", Verdict.SUPPORTED)),
           _verdicts(("Good matchup.", Verdict.SUPPORTED))],
          [""])
    out = run([PACKET], {"Josh Allen"})
    assert out.sections[0].prose == ("Good matchup.",)
    assert out.cut[0].outcome == "dropped"


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


# ------------------------------------------------- running out of the day's allowance

def test_quota_exhaustion_ships_the_lineup_without_the_commentary(monkeypatch):
    """The free tier is 20 requests a day per model. The lineup, the injury filter and
    every printed figure need no model at all, so that half still goes out."""
    def boom(*a, **k):
        raise QuotaExhausted("429 RESOURCE_EXHAUSTED, limit 20/day")

    monkeypatch.setattr(pipeline, "write", boom)
    out = run([PACKET], {"Josh Allen"})

    assert out.sections[0].prose == ()
    assert out.sections[0].templated                  # the figures survive
    assert "quota" in out.fallback_reason


def test_prose_written_but_never_audited_is_not_shipped(monkeypatch):
    """The writer succeeded and the auditor did not. An unchecked claim is the single
    thing this pipeline exists to stop, so the sentences move to `dropped` - kept for
    inspection, kept out of the output."""
    def boom(*a, **k):
        raise QuotaExhausted("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(pipeline, "write", lambda p, s, m: [_section("Unchecked claim.")])
    monkeypatch.setattr(pipeline, "audit_roster", boom)
    out = run([PACKET], {"Josh Allen"})

    assert out.sections[0].prose == ()
    assert "Unchecked claim." in out.sections[0].dropped
    assert out.fallback_reason
