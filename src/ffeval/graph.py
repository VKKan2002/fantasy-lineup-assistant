"""The rewrite loop as a LangGraph state machine.

The only copy of the loop. It used to be a for-loop in pipeline.run(); that was deleted
once this matched it, so there is one spelling to maintain, not two to keep in step.

Why a graph here and nowhere else: write -> audit -> rewrite -> audit again is a cycle with
a budget and a terminal condition, which is the one shape a plain function makes awkward
and a graph makes obvious. Everything upstream of it (packets, projection, best_lineup)
is a straight line and stays a straight line - see docs/DESIGN.md, "a workflow, not an
agent". The helpers the nodes call (_flagged, _apply, Rewrite) stay in pipeline.py.

A failed model call is a route, not an exception. Every model-calling node is wrapped so
any failure - quota gone, a 503 "high demand", a 413 "too large", a reply that is not JSON
- lands in `fallback_reason`, and the next edge sends the run to the fallback node rather
than up the stack. The run is unattended on a Sunday: a crash means no email and no clue.
Prose written but never audited is dropped there, because an unchecked claim is the one
thing this pipeline exists to stop.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .audit.auditor import QuotaExhausted
from .audit.packet import FactsPacket
from .pipeline import AUDITOR_MODEL, BUDGET, Rewrite, _apply, _flagged
from .writer import MODEL as WRITER_MODEL
from .writer import PlayerSection, facts_only, rewrite, write


class State(TypedDict, total=False):
    """What flows between nodes.

    `changed` is None on the first audit (judge everyone) and a set afterwards (judge only
    the players whose notes actually moved). Re-asking about an unchanged section cannot
    produce a new verdict and would just spend quota.
    """

    packets: list[FactsPacket]
    starters: set[str]
    writer_model: str
    auditor_model: str

    sections: list[PlayerSection]
    history: list[Rewrite]
    flagged: list[tuple]          # [(index, packet, sentence, verdict, reason)]
    changed: set[int] | None
    round: int
    fallback_reason: str


def _guard(fn: Callable[[State], dict]) -> Callable[[State], dict]:
    """Turn a model node's failure into state instead of a stack trace.

    Deliberately broad. On one day the run died on two 503s and a 413, none of them
    quota; each would have cost the whole email. The reason is recorded and printed, so a
    real bug still shows up - as a figures-only email that says why, not as silence.

    Also makes a node a no-op once a failure is recorded, so a route that has already been
    decided cannot be undone by a later node spending a request it does not have.
    """
    def node(state: State) -> dict:
        if state.get("fallback_reason"):
            return {}
        try:
            return fn(state)
        except QuotaExhausted as e:
            return {"fallback_reason":
                    f"daily model quota exhausted; lineup and figures only - {e}"}
        except Exception as e:
            return {"fallback_reason": f"model call failed ({type(e).__name__}); "
                                       f"lineup and figures only - {str(e)[:200]}"}
    return node


@_guard
def write_notes(state: State) -> dict:
    sections = list(write(state["packets"], state["starters"],
                          state.get("writer_model", WRITER_MODEL)))
    return {"sections": sections, "history": [], "changed": None, "round": 0}


@_guard
def audit(state: State) -> dict:
    """Judge the sections that could have changed. One model call for the whole roster."""
    flagged = _flagged(state["packets"], state["sections"],
                       state.get("auditor_model", AUDITOR_MODEL), only=state["changed"])
    return {"flagged": flagged, "round": state["round"] + 1}


@_guard
def rewrite_bad(state: State) -> dict:
    """Send back only the rejected sentences, and record what came back.

    Rewriting a whole player's notes would let a sentence that already passed come back
    worse, and no score could then say whether the loop helped.
    """
    flagged = state["flagged"]
    replacements = rewrite([(p, s, why) for _, p, s, _, why in flagged],
                           state.get("writer_model", WRITER_MODEL))

    sections = list(state["sections"])
    history = list(state["history"])
    changed: set[int] = set()
    swaps: dict[int, dict[str, str]] = {}
    for (i, _, sentence, verdict, why), new in zip(flagged, replacements):
        swaps.setdefault(i, {})[sentence] = new
        changed.add(i)
        history.append(Rewrite(
            player=sections[i].player, round=state["round"], original=sentence,
            verdict=verdict, reason=why, replacement=new,
            outcome="replaced" if new else "dropped",
        ))
    for i, swap in swaps.items():
        sections[i] = _apply(sections[i], swap)
    return {"sections": sections, "history": history, "changed": changed}


def cut(state: State) -> dict:
    """Budget spent. Anything still unfaithful is deleted rather than shipped.

    No model call, so no guard: this node only removes sentences the last audit already
    judged.
    """
    sections = list(state["sections"])
    history = list(state["history"])
    for i, _, sentence, verdict, why in state["flagged"]:
        sections[i] = _apply(sections[i], {sentence: ""})
        history.append(Rewrite(
            player=sections[i].player, round=state["round"], original=sentence,
            verdict=verdict, reason=why, replacement="", outcome="deleted",
        ))
    return {"sections": sections, "history": history}


def fallback(state: State) -> dict:
    """A model call failed. Ship the half that never needed a model.

    Prose that was written but never audited moves to `dropped`. "No notes today" is a
    better email than an unverified one.
    """
    sections = state.get("sections") or list(
        facts_only(state["packets"], state["starters"]))
    return {"sections": [replace(s, prose=(), dropped=s.dropped + s.prose)
                         for s in sections],
            "history": state.get("history", [])}


def after_audit(state: State) -> str:
    """The only branch in the graph: rewrite, cut, give up, or stop."""
    if state.get("fallback_reason"):
        return "fallback"
    if not state["flagged"]:
        return END
    return "rewrite_bad" if state["round"] <= BUDGET else "cut"


def _after_model(state: State) -> str:
    return "fallback" if state.get("fallback_reason") else "audit"


def build() -> StateGraph:
    g = StateGraph(State)
    for name, fn in (("write_notes", write_notes), ("audit", audit),
                     ("rewrite_bad", rewrite_bad), ("cut", cut), ("fallback", fallback)):
        g.add_node(name, fn)

    g.add_edge(START, "write_notes")
    g.add_conditional_edges("write_notes", _after_model, ["audit", "fallback"])
    g.add_conditional_edges("audit", after_audit,
                            ["rewrite_bad", "cut", "fallback", END])
    g.add_conditional_edges("rewrite_bad", _after_model, ["audit", "fallback"])
    g.add_edge("cut", END)
    g.add_edge("fallback", END)
    return g.compile()


GRAPH = build()
