"""Turn facts packets into the prose a manager reads.

The model never types a number. Every figure in the output is rendered from a fact by
_line() below; the model is given the same facts and asked for judgement and news only,
and any sentence it returns carrying a digit is dropped rather than trusted. A figure
that cannot be typed cannot be mistyped, which retires most of the failure modes the
auditor exists to catch - the auditor stays as the backstop, not the only defence.

The two kinds of sentence are kept apart all the way out. Auditing our own templated
lines against our own facts proves nothing and would flatter the score; the number worth
measuring is how often the MODEL invents something, and that needs its sentences alone.

One call per roster, not per player: a full roster fits in context, and per-player calls
would mean hundreds of requests arriving the moment cron fires.

Nothing personal goes in the prompt. Players are public figures and appear by name; the
manager's name, team name and address never reach the model - the email is assembled in
our own code afterwards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .audit.auditor import call_model, split_claims, strip_fence
from .audit.packet import NUMBER, Fact, FactsPacket

MODEL = "gemma-4-31b-it"   # writer != auditor, deliberately. See docs/DESIGN.md.

_RULES = """You are writing weekly fantasy football notes for one manager.

For each player you are given a facts packet and whether the lineup starts or sits them.
The lineup decision is already made and is not yours to change. If you disagree, say so
in words.

Rules:
1. NEVER write a digit. Not one. The numbers are printed separately by the program that
   calls you. Write "a strong outing", never "38.76 points". A sentence containing any
   digit will be thrown away.
2. Say where news came from. "The coach said Tuesday..." is fine. Stating the same thing
   as fact - "he is expected to practise normally" - is not, because a report supports
   that someone SAID something, not the thing itself.
3. If a news item is from an earlier season, ignore it. The current facts win.
4. Two or three sentences per player. No preamble, no sign-off.
5. If there is nothing worth saying, say that plainly."""

_OUTPUT = """Reply with ONLY a JSON array, no prose and no code fence, one object per
player, in the order given:

[{"player": "Josh Allen", "notes": "..."}]"""


@dataclass(frozen=True)
class PlayerSection:
    """One player's block of the email.

    `templated` and `prose` are separate because only `prose` is worth auditing - see
    the module docstring. `dropped` is the model's rule-breaking output, kept rather
    than discarded so a prompt that has started emitting digits is visible instead of
    quietly shrinking.
    """

    player: str
    decision: str                   # "start" or "sit"
    templated: tuple[str, ...]      # ours. every number in the output lives here
    prose: tuple[str, ...]          # the model's. the only sentences to audit
    dropped: tuple[str, ...]        # model sentences refused for carrying a digit


def _line(fact: Fact) -> str:
    """One fact -> one sentence. The label already reads as English, so use it.

    ponytail: no per-fact template table. Add one when a label reads badly in a real
    email - which is also when someone will want to drop the dull facts entirely.
    """
    label = fact.label.split(" (")[0]
    value = fact.value
    if isinstance(value, bool):
        value = "yes" if value else "no"
    return f"{label}: {value}"


def build_prompt(packets: list[FactsPacket], starters: set[str]) -> str:
    """Rules, then every player's packet, then the output format.

    packet.render() is reused as-is, so the news arrives inside the same untrusted-input
    markers the auditor sees. One prompt, one call, whole roster.
    """
    blocks = []
    for p in packets:
        verdict = "START" if p.player in starters else "SIT"
        blocks.append(f"--- {p.player} ({verdict}) ---\n{p.render()}")
    return f"{_RULES}\n\n" + "\n\n".join(blocks) + f"\n\n{_OUTPUT}\n"


def _sections(
    packets: list[FactsPacket], starters: set[str], notes: dict[str, str]
) -> list[PlayerSection]:
    """Assemble the output. Pure - no model, no network, so it is testable without a key."""
    out = []
    for p in packets:
        # split_claims, not str.split("."), or "38.76" becomes two sentences - and the
        # auditor splits the same way, so a sentence here is the unit it will judge.
        keep, dropped = [], []
        for sentence in split_claims(notes.get(p.player, "")):
            if NUMBER.search(sentence):
                dropped.append(sentence)
            else:
                keep.append(sentence)
        out.append(
            PlayerSection(
                player=p.player,
                decision="start" if p.player in starters else "sit",
                templated=tuple(_line(f) for f in p.facts),
                prose=tuple(keep),
                dropped=tuple(dropped),
            )
        )
    return out


def write(
    packets: list[FactsPacket], starters: set[str], model: str = MODEL
) -> list[PlayerSection]:
    """Facts in, one section per player out. One model call for the whole roster."""
    raw = call_model(build_prompt(packets, starters), model)
    rows = json.loads(strip_fence(raw))
    return _sections(packets, starters, {r["player"]: r.get("notes", "") for r in rows})
