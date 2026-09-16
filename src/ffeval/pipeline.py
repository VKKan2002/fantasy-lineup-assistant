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

from .audit.auditor import QuotaExhausted, audit_roster
from .audit.packet import FactsPacket
from .audit.verdicts import UNFAITHFUL
from .models.expected import project_ppg
from .scoring.league import League
from .scoring.lineup import best_lineup
from .writer import MODEL as WRITER_MODEL
from .writer import PlayerSection, facts_only, rewrite, write

AUDITOR_MODEL = "gemini-3.6-flash"   # not the writer's. See docs/DESIGN.md.
BUDGET = 2                           # rewrite rounds before a sentence is cut


def _can_play(packet: FactsPacket) -> bool:
    """Is this player eligible to be in a lineup at all?

    best_lineup() only ever sees a number, so anyone who physically cannot play has to be
    removed before it runs. A player ruled Out has a fine projection - good draft position,
    good history - and will otherwise be started, which is how CeeDee Lamb was slotted in
    beside a note saying he was out for the week.

    Only "Out" blocks. Doubtful and Questionable are probabilities, not facts, and this
    project's rule is that the projection handles uncertainty while the model comments on
    it - neither of them gets to overrule the lineup.
    """
    for f in packet.facts:
        if f.id == "bye.is_bye_week" and f.value:
            return False
        if f.id == "injury.report_status" and str(f.value).lower() == "out":
            return False
    return True


def decide_starters(
    packets: list[FactsPacket],
    priors: dict[str, float],
    history: dict[str, list[float]],
    league: League | None = None,
) -> tuple[set[str], dict[str, float | None]]:
    """Who to start this week, and what each player was projected at.

    The model does not appear anywhere in this function, and that is the design rather
    than an omission: best_lineup() is provably optimal given projections, and a
    twelve-line rule already captures 91.2% of the available points against 94.0% for a
    cheater who knows everything. There is nothing here for a model to discover.

    A player the rule cannot rank - undrafted and yet to play - is left out of the lineup
    rather than projected at zero. Zero is a claim; None is the truth. They come back in
    the projections mapping so the caller can say who was skipped and why.

    A player on a bye is excluded too. best_lineup() would happily start him otherwise,
    since it only sees a number.
    """
    league = league or League()
    projections: dict[str, float | None] = {}
    available = []
    for p in packets:
        proj = project_ppg(history.get(p.player_id, []), priors.get(p.player_id))
        projections[p.player] = proj
        if proj is not None and _can_play(p):
            available.append((p.player, p.position, proj))

    _, chosen = best_lineup(available, league)
    return {name for name, _, _ in chosen}, projections


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
    # Empty when the run completed. Set when the day's quota ran out and the commentary
    # was dropped - the lineup and the figures still went out, because neither needs a
    # model. Unaudited prose never ships.
    fallback_reason: str = ""

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
    todo = [(i, p, list(s.prose)) for i, (p, s) in enumerate(zip(packets, sections))
            if s.prose and (only is None or i in only)]
    if not todo:
        return []

    out = []
    results = audit_roster([(p, claims) for _, p, claims in todo], model)
    for (i, packet, _), result in zip(todo, results):
        for v in result.verdicts:
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
    """Facts in, audited notes out. At most BUDGET rewrite calls for the whole roster.

    Runs out of quota gracefully. The free tier grants 20 requests a day per model, and
    the lineup, the injury filter and every printed figure need no model at all - so when
    the allowance is gone that half still goes out, and the commentary does not. Prose
    that was written but never audited is moved to `dropped` rather than shipped: an
    unchecked claim is the one thing this pipeline exists to stop.
    """
    sections: list[PlayerSection] = []
    history: list[Rewrite] = []
    try:
        sections = list(write(packets, starters, writer_model))
        return _run_loop(packets, sections, history, writer_model, auditor_model)
    except QuotaExhausted as e:
        if not sections:
            sections = list(facts_only(packets, starters))
        return Result(
            sections=tuple(replace(s, prose=(), dropped=s.dropped + s.prose)
                           for s in sections),
            rewrites=tuple(history),
            fallback_reason=f"daily model quota exhausted; lineup and figures only - {e}",
        )


def _run_loop(
    packets: list[FactsPacket],
    sections: list[PlayerSection],
    history: list[Rewrite],
    writer_model: str,
    auditor_model: str,
) -> Result:
    """The audit/rewrite rounds. Split out so run() can wrap the whole thing in one
    quota guard without burying the loop in a try block."""
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
