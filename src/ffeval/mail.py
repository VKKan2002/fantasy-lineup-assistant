"""The report as an email: HTML with inline styles, a plain-text copy, one HTTPS request.

No libraries (Forks 21a, 22a). Email clients drop stylesheets and scripts, so every style
sits on its element and the layout is a table - the one thing every client renders the
same. The plain-text copy is report.render(), so the two can never say different things.

Everything that came from outside - player names, news, model prose - goes through
html.escape. A note containing "<" must arrive as text, not as markup.

Sent through Resend's testing sender, which only delivers to the account's own address.
That is the "only me for now" stage; other managers need a verified domain.
"""

from __future__ import annotations

import json
import os
import urllib.request
from html import escape

from .audit.packet import FactsPacket
from .report import TOSS_UP, summary
from .writer import PlayerSection

_FONT = "font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;"
_BADGE = {"Questionable": "#b45309", "Doubtful": "#b91c1c", "Out": "#b91c1c"}


def html_report(week: int, pairs, starters: set[str], projections: dict[str, float | None],
                sections: list[PlayerSection], notes: list[str], status: dict[str, str],
                fallback: str = "", packets: dict[str, FactsPacket] | None = None) -> str:
    e = escape
    rows = []
    if fallback:
        rows.append(f'<p style="background:#fef3c7;padding:10px 12px;border-radius:6px;'
                    f'color:#78350f;font-size:16px">Notes are off this week: {e(fallback)}</p>')

    # the swaps: the one thing to act on, so it goes first and gets colour
    if pairs:
        items = []
        for sit, start, gain in pairs:
            if gain is None:
                g, colour = "gain unknown", "#374151"
            elif gain < TOSS_UP:
                g, colour = f"{gain:+.1f} &middot; toss-up, your call", "#6b7280"
            else:
                g, colour = f"{gain:+.1f} projected", "#15803d"
            items.append(
                f'<tr><td style="padding:8px 0;border-bottom:1px solid #d1fae5">'
                f'<b style="color:#15803d;font-size:17px">&#9650; Start {e(start)}</b><br>'
                f'<span style="color:#4b5563;font-size:16px">&#9660; Sit {e(sit)}</span></td>'
                f'<td style="padding:8px 0;text-align:right;color:{colour};'
                f'border-bottom:1px solid #d1fae5;white-space:nowrap;font-size:16px">{g}</td></tr>')
        rows.append('<div style="background:#ecfdf5;border-radius:8px;padding:6px 14px">'
                    f'<table width="100%" style="border-collapse:collapse">{"".join(items)}'
                    '</table></div>')
    else:
        rows.append('<p style="background:#f3f4f6;padding:12px;border-radius:8px">'
                    "<b>Nothing I'd change.</b></p>")

    # the lineup
    lineup = "".join(
        f'<tr><td style="padding:5px 0">{e(s.player)}</td><td style="text-align:right;'
        f'color:#374151">{"" if projections.get(s.player) is None else f"{projections[s.player]:.1f}"}'
        f'</td></tr>' for s in sections if s.player in starters)
    rows.append('<h3 style="margin:22px 0 6px;font-size:14px;letter-spacing:.06em;'
                'color:#6b7280">RECOMMENDED LINEUP &middot; PROJECTED</h3>'
                f'<table width="100%" style="border-collapse:collapse;font-size:17px">{lineup}'
                '</table>')

    # one card per player: the checked notes first, then a few plain-English facts
    for s in sections:
        st = status.get(s.player, "none")
        badge = (f' <span style="background:{_BADGE[st]};color:#fff;font-size:12px;'
                 f'padding:2px 6px;border-radius:4px">{e(st)}</span>' if st in _BADGE else "")
        verdict = "#15803d" if s.decision == "start" else "#6b7280"
        prose = "".join(f'<p style="margin:8px 0;font-size:17px;line-height:1.45">'
                        f'{e(t)}</p>' for t in s.prose)
        shown = summary(packets[s.player]) if packets and s.player in packets else s.templated
        facts = "<br>".join(e(t) for t in shown)
        rows.append(
            '<div style="border:1px solid #e5e7eb;border-radius:8px;padding:12px 14px;'
            'margin-top:12px">'
            f'<div style="font-size:18px"><b>{e(s.player)}</b> <span style="color:{verdict};'
            f'font-size:14px">{s.decision.upper()}</span>{badge}</div>{prose}'
            f'<div style="color:#4b5563;font-size:15px;margin-top:8px;line-height:1.6">'
            f'{facts}</div></div>')

    if notes:
        rows.append('<p style="color:#4b5563;font-size:15px;margin-top:18px">'
                    + "<br>".join(e(n) for n in notes) + "</p>")

    return (f'<div style="{_FONT}max-width:600px;margin:0 auto;padding:16px;'
            f'color:#111827;font-size:16px;background:#ffffff"><h2 style="margin:0 0 14px;font-size:26px">Week {week}</h2>'
            + "".join(rows) + "</div>")


def send(subject: str, html: str, text: str) -> str:
    """One POST to Resend. Returns the message id. Needs RESEND_API_KEY and EMAIL_TO."""
    from dotenv import load_dotenv

    load_dotenv()
    body = json.dumps({"from": "Fantasy report <onboarding@resend.dev>",
                       "to": [os.environ["EMAIL_TO"]], "subject": subject,
                       "html": html, "text": text}).encode()
    req = urllib.request.Request(
        "https://api.resend.com/emails", data=body, method="POST",
        headers={"Authorization": f"Bearer {os.environ['RESEND_API_KEY']}",
                 "Content-Type": "application/json", "User-Agent": "ffeval"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["id"]
