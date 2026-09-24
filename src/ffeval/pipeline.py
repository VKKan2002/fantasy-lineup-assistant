"""The pieces the rewrite loop is built from. The loop itself is in graph.py.

Write, audit, rewrite what failed, and delete whatever never came clean.

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

from .audit.auditor import audit_roster
from .audit.packet import FactsPacket
from .audit.verdicts import UNFAITHFUL
from .models.expected import project_ppg
from .scoring.league import League
from .scoring.lineup import best_lineup
from .writer import PlayerSection

AUDITOR_MODEL = "groq/openai/gpt-oss-120b"   # not the writer's. See docs/DESIGN.md.
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
    must_start: set[str] = frozenset(),
    cannot_start: set[str] = frozenset(),
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

    must_start / cannot_start are player names whose game has kicked off: ESPN has locked
    them where they are. A locked starter is in the lineup whatever his projection, and a
    locked bench player cannot be moved in. Recommending either change is advice nobody
    can take.
    """
    league = league or League()
    projections: dict[str, float | None] = {}
    available = []
    for p in packets:
        proj = project_ppg(history.get(p.player_id, []), priors.get(p.player_id))
        projections[p.player] = proj
        if p.player in must_start:
            # ponytail: a huge score makes best_lineup take him first at his position. The
            # real projection is still what `projections` reports.
            available.append((p.player, p.position, 1e9))
        elif proj is not None and _can_play(p) and p.player not in cannot_start:
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
