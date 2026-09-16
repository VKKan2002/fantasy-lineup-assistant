"""Tests for the rewrite loop. The model calls are faked; the loop's decisions are not.

What matters here is termination and what survives: that a clean pass costs nothing, that
a sentence which never comes clean is cut rather than shipped, and that the budget is
actually a bound.
"""

from pathlib import Path

import ffeval.pipeline as pipeline
from ffeval.audit.packet import FactsPacket
from ffeval.audit.verdicts import AuditResult, ClaimVerdict, Verdict
from ffeval.pipeline import BUDGET, _apply, run
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

    def fake_audit(packet, claims, model):
        calls["audit"] += 1
        return audits.pop(0)

    def fake_rewrite(items, model=None):
        calls["rewrite"] += 1
        return [replacements.pop(0) for _ in items]

    monkeypatch.setattr(pipeline, "write", lambda p, s, m: list(sections))
    monkeypatch.setattr(pipeline, "audit_claims", fake_audit)
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
