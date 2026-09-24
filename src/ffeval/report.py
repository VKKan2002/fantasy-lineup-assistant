"""The weekly report as plain text (Fork 20a). No model, no network: every line here is
the lineup decision or a section the rewrite loop already checked.

Swaps carry their projected gain. Under 1 point is labelled a toss-up (Fork 19a):
perfect information is worth 2.9 points a week in total, so a swap gaining less than one
is mostly noise, and saying so is the honest version of recommending it.
"""

from __future__ import annotations

from .audit.packet import FactsPacket
from .scoring.league import FLEX_ELIGIBLE
from .writer import PlayerSection

TOSS_UP = 1.0   # points; see the module docstring


def _nth(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def summary(p: FactsPacket) -> list[str]:
    """The facts a person reads, in a few short lines. The packet's labels were written
    for the model - "official injury report status: Out, Doubtful, Questionable, or none
    if not listed: none" - and a manager does not need twelve of them. The model still
    gets every fact; this only decides what the email shows. Injury and bye lines appear
    only when they say something.
    """
    f = {x.id: x.value for x in p.facts}
    out = []
    games = [round(v, 1) for k, v in sorted(f.items()) if k.startswith("form.game_w")]
    if games:
        out.append("Last games: " + ", ".join(f"{g:g}" for g in games))
    elif "espn.points_per_game_this_season" in f:
        last = f.get("espn.points_per_game_last_season")
        out.append(f"Per game: {f['espn.points_per_game_this_season']:g} this season"
                   + ("" if last is None else f", {last:g} last season"))
    if f.get("bye.is_bye_week"):
        return out + ["On a bye this week"]
    if "matchup.spread_line" in f:
        spread = f["matchup.spread_line"]
        odds = ("even" if spread == 0 else
                f"favored by {spread:g}" if spread > 0 else f"underdog by {-spread:g}")
        where = "home" if f.get("matchup.is_home") else "away"
        out.append(f"vs {p.opponent} ({where}) · {odds} · total {f['matchup.total_line']:g}")
    pos = p.position.lower()
    rank = f.get(f"defense.{pos}_ppr_allowed_rank")
    if rank is not None:
        per = f.get(f"defense.{pos}_ppr_allowed_per_game")
        out.append(f"{p.opponent} allow the {_nth(int(rank))}-most points to {p.position}s"
                   + ("" if per is None else f" ({per:.1f} a game)"))
    status = str(f.get("injury.report_status", "none"))
    practice = str(f.get("injury.practice_status", "none"))
    if status != "none" or practice != "none":
        out.append(f"Injury report: {status}" + ("" if practice == "none" else f" · {practice}"))
    return out


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
           sections: list[PlayerSection], notes: list[str], fallback: str = "",
           packets: dict[str, FactsPacket] | None = None) -> str:
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
        facts = summary(packets[s.player]) if packets and s.player in packets else s.templated
        lines += [f"  {t}" for t in facts]
        lines += [f"  > {t}" for t in s.prose]
    if notes:
        lines += ["", *notes]
    return "\n".join(lines)
