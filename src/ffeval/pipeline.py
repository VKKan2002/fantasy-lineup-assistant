"""Write, audit, rewrite what failed, and delete whatever never came clean.

This is the loop docs/DESIGN.md calls genuinely agentic: it runs until every claim is
grounded or the budget is spent, terminating on a verifiable condition rather than a fixed
number of steps. Two rounds, then the survivors are cut.

It is the first code that owns both the writer and the auditor, which is why it is neither
of them. A writer that ran its own grading loop would be deciding when it had passed.

Only the sentences the auditor rejected are sent back. Rewriting a whole player's notes
would let a sentence that already passed come back worse, and no score could then say
whether the loop helped. The budget is per ROSTER, not per sentence: one model call per
round however many sentences failed, because the design's one-call-per-manager rule exists
to keep a burst of requests off a per-minute rate limit.

What was rejected, what replaced it, and what was cut are all kept. A loop that silently
deletes looks identical to a loop with nothing to do, and the difference is exactly what
tells you a prompt has started to rot.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .audit.auditor import audit_claims
from .audit.packet import FactsPacket
from .audit.verdicts import UNFAITHFUL
from .writer import MODEL as WRITER_MODEL
from .writer import PlayerSection, rewrite, write

AUDITOR_MODEL = "gemini-3.6-flash"   # not the writer's. See docs/DESIGN.md.
BUDGET = 2                           # rewrite rounds before a sentence is cut


@dataclass(frozen=True)
class Rewrite:
    """One rejected sentence and what happened to it. A log entry, not a verdict.

    A replacement that fails again gets its own entry next round, so the history reads
    as a sequence of events rather than a single outcome per sentence.
    """

    player: str
    round: int
    original: str
    verdict: str
    reason: str
    replacement: str      # "" when the writer had nothing supportable to offer
    outcome: str          # "replaced", "dropped" (writer gave up), "deleted" (budget out)


@dataclass(frozen=True)
class Result:
    sections: tuple[PlayerSection, ...]
    rewrites: tuple[Rewrite, ...]

    @property
    def cut(self) -> tuple[Rewrite, ...]:
        """Everything that never came clean - dropped by the writer or cut at the budget."""
        return tuple(r for r in self.rewrites if r.outcome in ("dropped", "deleted"))


def _flagged(
    packets: list[FactsPacket],
    sections: list[PlayerSection],
    model: str,
    only: set[int] | None = None,
) -> list[tuple[int, FactsPacket, str, str, str]]:
    """Audit and return [(index, packet, sentence, verdict, reason)] for the bad ones.

    `only` limits the re-audit to players whose notes actually changed; the rest cannot
    have new verdicts and re-asking would just spend quota.
    """
    out = []
    for i, (packet, section) in enumerate(zip(packets, sections)):
        if not section.prose or (only is not None and i not in only):
            continue
        for v in audit_claims(packet, list(section.prose), model).verdicts:
            if v.verdict in UNFAITHFUL:
                out.append((i, packet, v.claim, v.verdict.value, v.reason))
    return out


def _apply(section: PlayerSection, swaps: dict[str, str]) -> PlayerSection:
    """Swap rejected sentences for their replacements; an empty replacement removes it."""
    prose = tuple(swaps.get(s, s) for s in section.prose)
    return replace(section, prose=tuple(s for s in prose if s))


def run(
    packets: list[FactsPacket],
    starters: set[str],
    writer_model: str = WRITER_MODEL,
    auditor_model: str = AUDITOR_MODEL,
) -> Result:
    """Facts in, audited notes out. At most BUDGET rewrite calls for the whole roster."""
    sections = list(write(packets, starters, writer_model))
    history: list[Rewrite] = []
    changed: set[int] | None = None

    for rnd in range(1, BUDGET + 1):
        bad = _flagged(packets, sections, auditor_model, only=changed)
        if not bad:
            return Result(tuple(sections), tuple(history))

        replacements = rewrite([(p, s, why) for _, p, s, _, why in bad], writer_model)
        changed = set()
        swaps: dict[int, dict[str, str]] = {}
        for (i, _, sentence, verdict, why), new in zip(bad, replacements):
            swaps.setdefault(i, {})[sentence] = new
            changed.add(i)
            history.append(Rewrite(
                player=sections[i].player, round=rnd, original=sentence,
                verdict=verdict, reason=why, replacement=new,
                outcome="replaced" if new else "dropped",
            ))
        for i, swap in swaps.items():
            sections[i] = _apply(sections[i], swap)

    # Budget spent. Whatever the last round produced is judged once more, and anything
    # still unfaithful is cut - shipping it is the single outcome this loop exists to
    # prevent, and a second chance already came and went.
    for i, _, sentence, verdict, why in _flagged(
        packets, sections, auditor_model, only=changed
    ):
        sections[i] = _apply(sections[i], {sentence: ""})
        history.append(Rewrite(
            player=sections[i].player, round=BUDGET + 1, original=sentence,
            verdict=verdict, reason=why, replacement="", outcome="deleted",
        ))
    return Result(tuple(sections), tuple(history))
