"""Tests for the rewrite loop. The model calls are faked; the routing is not.

What matters is termination and what survives: a clean pass costs nothing, a sentence that
never comes clean is cut rather than shipped, the budget is a real bound, and running out
of quota ships the figures without the prose.
"""

import ffeval.graph as graph
import ffeval.pipeline as pipeline
from ffeval.audit.auditor import QuotaExhausted
from ffeval.pipeline import BUDGET
from test_pipeline import PACKET, _section, _verdicts
from ffeval.audit.verdicts import Verdict


def _fake(monkeypatch, sections, audits, replacements):
    """Same wiring as test_pipeline._fake, pointed at the graph's namespace."""
    calls = {"audit": 0, "rewrite": 0}

    def fake_audit(items, model):
        calls["audit"] += 1
        return [audits.pop(0) for _ in items]

    def fake_rewrite(items, model=None):
        calls["rewrite"] += 1
        return [replacements.pop(0) for _ in items]

    monkeypatch.setattr(graph, "write", lambda p, s, m: list(sections))
    monkeypatch.setattr(graph, "rewrite", fake_rewrite)
    monkeypatch.setattr(pipeline, "audit_roster", fake_audit)   # _flagged lives there
    return calls


def _invoke(**over):
    state = {"packets": [PACKET], "starters": {"Josh Allen"},
             "writer_model": "fake", "auditor_model": "fake"}
    return graph.GRAPH.invoke(state | over)


BAD = ("He threw for 900 yards.", Verdict.CONTRADICTED)
GOOD = ("He is a strong play.", Verdict.SUPPORTED)


def test_clean_pass_never_calls_the_writer_back(monkeypatch):
    calls = _fake(monkeypatch, [_section(GOOD[0])], [_verdicts(GOOD)], [])
    out = _invoke()
    assert calls["audit"] == 1 and calls["rewrite"] == 0
    assert out["sections"][0].prose == (GOOD[0],)
    assert out["history"] == []


def test_a_sentence_that_never_comes_clean_is_cut(monkeypatch):
    """Budget rounds of rewriting, one last judgement, then the sentence is deleted."""
    calls = _fake(monkeypatch,
                  [_section(BAD[0])],
                  [_verdicts(BAD) for _ in range(BUDGET + 1)],
                  [BAD[0]] * BUDGET)
    out = _invoke()
    assert calls["rewrite"] == BUDGET
    assert calls["audit"] == BUDGET + 1          # the extra one is the final judgement
    assert out["sections"][0].prose == ()        # cut, not shipped
    assert [r.outcome for r in out["history"]] == ["replaced"] * BUDGET + ["deleted"]


def test_quota_exhausted_drops_prose_and_keeps_the_figures(monkeypatch):
    """An unaudited sentence never ships, but the lineup needs no model and still does."""
    _fake(monkeypatch, [_section(GOOD[0])], [], [])
    monkeypatch.setattr(pipeline, "audit_roster",
                        lambda i, m: (_ for _ in ()).throw(QuotaExhausted("429")))
    out = _invoke()
    assert out["fallback_reason"].startswith("daily model quota exhausted")
    assert out["sections"][0].prose == ()
    assert out["sections"][0].dropped == (GOOD[0],)
    assert out["sections"][0].templated == ("a: 1",)


def test_an_empty_replacement_drops_the_sentence(monkeypatch):
    """The writer may say there is nothing supportable to write. That beats an invented
    replacement, so "" removes the sentence instead of blanking it."""
    other = ("Good matchup.", Verdict.SUPPORTED)
    _fake(monkeypatch, [_section(BAD[0], other[0])],
          [_verdicts(BAD, other), _verdicts(other)], [""])
    out = _invoke()
    assert out["sections"][0].prose == (other[0],)
    assert out["history"][0].outcome == "dropped"


def test_quota_gone_before_writing_still_ships_the_figures(monkeypatch):
    """No sections exist yet, so the fallback has to build the figures-only version."""
    _fake(monkeypatch, [], [], [])
    monkeypatch.setattr(graph, "write",
                        lambda *a: (_ for _ in ()).throw(QuotaExhausted("429")))
    out = _invoke()
    assert "quota" in out["fallback_reason"]
    assert out["sections"][0].prose == ()
    assert out["sections"][0].templated                  # the figures survive


def test_any_model_failure_still_ships_the_figures(monkeypatch):
    """A 503 is not a quota error, and on a Sunday it must not cost the whole email."""
    _fake(monkeypatch, [_section(GOOD[0])], [], [])
    monkeypatch.setattr(pipeline, "audit_roster",
                        lambda i, m: (_ for _ in ()).throw(RuntimeError("503 UNAVAILABLE")))
    out = _invoke()
    assert "RuntimeError" in out["fallback_reason"]
    assert out["sections"][0].prose == ()
    assert out["sections"][0].templated
