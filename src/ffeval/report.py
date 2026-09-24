"""The weekly report as plain text (Fork 20a). No model, no network: every line here is
the lineup decision or a section the rewrite loop already checked.

Swaps carry their projected gain. Under 1 point is labelled a toss-up (Fork 19a):
perfect information is worth 2.9 points a week in total, so a swap gaining less than one
is mostly noise, and saying so is the honest version of recommending it.
"""

from __future__ import annotations

from .scoring.league import FLEX_ELIGIBLE
from .writer import PlayerSection

TOSS_UP = 1.0   # points; see the module docstring


def swaps(current: set[str], recommended: set[str], position: dict[str, str],
          value: dict[str, float | None]) -> list[tuple[str, str, float | None]]:
    """[(sit, start, projected gain)], pairing same-position players first, then FLEX.

    `value` is what a player is expected to score this week: 0 for someone who cannot
    play, None for someone nobody can project - a gain against None is unknown, not zero.
    """
    outs = sorted(current - recommended)
    ins = sorted(recommended - current)
    def fits(i: str, o: str, same: bool) -> bool:
        if same:
            return position[i] == position[o]
        return position[i] in FLEX_ELIGIBLE and position[o] in FLEX_ELIGIBLE

    pairs = []
    for same in (True, False):
        for o in list(outs):
            match = next((i for i in ins if fits(i, o, same)), None)
            if match:
                outs.remove(o)
                ins.remove(match)
                a, b = value.get(match), value.get(o)
                pairs.append((o, match, None if a is None or b is None else round(a - b, 1)))
    return pairs


def render(week: int, pairs, starters: set[str], projections: dict[str, float | None],
           sections: list[PlayerSection], notes: list[str], fallback: str = "") -> str:
    lines = [f"WEEK {week}", ""]
    if fallback:
        lines += [f"Notes are off this week: {fallback}", ""]
    if not pairs:
        lines.append("Nothing I'd change.")
    for sit, start, gain in pairs:
        g = "gain unknown" if gain is None else f"{gain:+.1f} projected"
        tag = "  (toss-up, your call)" if gain is not None and gain < TOSS_UP else ""
        lines.append(f"Start {start}, sit {sit}: {g}{tag}")
    lines += ["", "LINEUP"]
    for s in sections:
        if s.player in starters:
            p = projections.get(s.player)
            lines.append(f"  {s.player:<24} {'' if p is None else f'{p:5.1f}'}")
    for s in sections:
        lines += ["", f"{s.player} - {s.decision.upper()}"]
        lines += [f"  {t}" for t in s.templated]
        lines += [f"  > {t}" for t in s.prose]
    if notes:
        lines += ["", *notes]
    return "\n".join(lines)
