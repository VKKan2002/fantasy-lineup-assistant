"""Tests for the writer. No model, no network - _sections() is pure on purpose."""

from pathlib import Path

from ffeval.audit.packet import FactsPacket
from ffeval.writer import _line, _sections, build_prompt

PACKET = FactsPacket.load(Path("eval/packets/2025_w03_allen.json"))
STARTERS = {"Josh Allen"}


def test_a_digit_from_the_model_is_refused_not_printed():
    """The rule the whole design rests on: the model does not type numbers.

    The kept sentence and the dropped one say the same thing. Only the one carrying a
    figure is refused, and it is kept in `dropped` so a prompt that starts emitting
    digits is visible rather than quietly producing shorter output.
    """
    notes = {"Josh Allen": "He looked sharp in the opener. He scored 38.76 points."}
    s = _sections([PACKET], STARTERS, notes)[0]

    assert s.prose == ("He looked sharp in the opener.",)
    assert s.dropped == ("He scored 38.76 points.",)


def test_numbers_come_from_the_facts():
    """Every figure in the output is rendered from a fact, so it cannot be mistyped."""
    s = _sections([PACKET], STARTERS, {})[0]
    assert "PPR points scored in week 1: 38.76" in s.templated
    assert s.prose == ()


def test_booleans_read_as_words():
    bye = next(f for f in PACKET.facts if f.id == "bye.is_bye_week")
    assert _line(bye) == "on a bye this week and therefore cannot be started: no"


def test_sit_and_start_are_both_labelled():
    assert _sections([PACKET], {"Josh Allen"}, {})[0].decision == "start"
    assert _sections([PACKET], set(), {})[0].decision == "sit"


def test_prompt_carries_the_decision_and_the_untrusted_news_markers():
    """The lineup call is stated, and news reaches the model inside the same guard
    markers the auditor sees - packet.render() is reused rather than reimplemented."""
    prompt = build_prompt([PACKET], STARTERS)
    assert "Josh Allen (START)" in prompt
    assert "BEGIN UNTRUSTED NEWS" in prompt
    assert "NEVER state a number" in prompt
    assert "spelled out in words" in prompt      # the loophole the first wording left
