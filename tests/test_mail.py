"""Tests for the HTML email. No network: send() is not called here."""

from ffeval.mail import html_report
from ffeval.writer import PlayerSection


def _s(name, decision, prose=()):
    return PlayerSection(player=name, decision=decision, templated=("a: 1",),
                         prose=prose, dropped=())


def test_outside_text_arrives_as_text_not_markup():
    """News and model prose come from outside. A "<" in them must not become HTML."""
    html = html_report(3, [], {"A"}, {"A": 10.0},
                       [_s("A", "start", ("He said <script>x</script> & more.",))], [], {})
    assert "<script>" not in html
    assert "&lt;script&gt;" in html and "&amp; more" in html


def test_swaps_injury_badge_and_toss_up_all_show():
    html = html_report(3, [("B", "A", 0.4)], {"A"}, {"A": 10.0},
                       [_s("A", "start"), _s("B", "sit")], [], {"A": "Questionable"})
    assert "Start A" in html and "Sit B" in html
    assert "toss-up" in html
    assert ">Questionable</span>" in html
