"""Tests for the news fetch and the digging agent. Recorded ESPN replies, a scripted model.

What matters: the fetch keeps only fresh notes about this player; "ESPN is down" is
visible rather than looking like a quiet week; the agent can only hand back items a tool
really returned; and the budget is a real bound.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import ffeval.ingest.news as news
from ffeval.audit.packet import Fact, FactsPacket

NOW = datetime(2026, 9, 23, 20, 0, tzinfo=timezone.utc)
FIX = Path("tests/fixtures")


def _replay(url):
    """Record and replay: the only network door, answered from disk."""
    name = "espn_player_4426515.json" if "players" in url else "espn_team_14.json"
    return json.loads((FIX / name).read_text())


def _packet(status="Questionable"):
    return FactsPacket(
        player="Puka Nacua", position="WR", team="LA", opponent="DEN", season=2026, week=3,
        facts=(Fact(id="injury.report_status", label="status", value=status, unit="text",
                    source="test", as_of="2026 week 3", direction="neutral"),),
        news=(), player_id="00-0039075")


def _call(i, name, args=None):
    return {"id": f"c{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


def test_fetch_keeps_the_newest_three_notes_about_him(monkeypatch):
    monkeypatch.setattr(news, "_get", _replay)
    items = news.player_notes(4426515, NOW)
    assert len(items) == 3
    assert [i.id for i in items] == ["news.espn_01", "news.espn_02", "news.espn_03"]
    assert items[0].published > items[1].published > items[2].published
    assert "Nacua" in items[0].text              # a note about him, not an article
    assert all(i.url.startswith("http") for i in items)


def test_espn_down_is_visible_not_a_quiet_week(monkeypatch):
    def down(url):
        raise OSError("connection refused")
    monkeypatch.setattr(news, "_get", down)
    monkeypatch.setattr(news, "espn_ids", lambda ids: {"00-0039075": 4426515})
    [p] = news.add_news([_packet(status="none")], NOW)
    assert p.news == ()
    assert p.fact("news.unavailable").value is True


def test_the_agent_hands_back_only_items_a_tool_returned(monkeypatch):
    """It names one real item and one it made up. Only the real one reaches the packet."""
    monkeypatch.setattr(news, "_get", _replay)
    turns = iter([
        [_call(1, "team_news")],
        [_call(2, "done", {"item_ids": ["news.dig_01", "news.dig_99"], "answered": True})],
    ])
    monkeypatch.setattr(news, "_chat", lambda m, msgs, force_done: {
        "role": "assistant", "content": "", "tool_calls": next(turns)})

    p = news.dig(_packet(), 4426515, NOW, model="fake")
    assert [n.id for n in p.news] == ["news.dig_01"]          # the invented id is dropped
    assert p.fact("news.status_search").value == "answered"
    assert "1 tool calls" in p.fact("news.status_search").source


def test_the_budget_is_a_real_bound(monkeypatch):
    """A model that never stops searching is made to answer after four calls."""
    monkeypatch.setattr(news, "_get", _replay)
    forced = []

    def endless(model, msgs, force_done):
        forced.append(force_done)
        call = (_call(len(forced), "done", {"item_ids": [], "answered": False})
                if force_done else _call(len(forced), "team_news"))
        return {"role": "assistant", "content": "", "tool_calls": [call]}

    monkeypatch.setattr(news, "_chat", endless)
    p = news.dig(_packet(), 4426515, NOW, model="fake")
    assert forced == [False] * news.BUDGET + [True]
    assert p.fact("news.status_search").value == "could not confirm"


def test_an_agent_crash_costs_the_answer_not_the_packet(monkeypatch):
    def boom(*a):
        raise RuntimeError("groq down")
    monkeypatch.setattr(news, "_chat", boom)
    p = news.dig(_packet(), 4426515, NOW, model="fake")
    assert p.fact("news.status_search").value.startswith("could not confirm")
    assert p.fact("injury.report_status").value == "Questionable"   # packet intact


def test_a_midweek_injury_note_opens_the_question(monkeypatch):
    """Wednesday: no official status yet, but the newest note says "Nacua (groin)" and
    "uncertain". That has to start the agent, or it never runs before Friday."""
    monkeypatch.setattr(news, "_get", _replay)
    monkeypatch.setattr(news, "espn_ids", lambda ids: {"00-0039075": 4426515})
    ran = []
    monkeypatch.setattr(news, "dig", lambda p, *a, **k: ran.append(p.player) or p)
    news.add_news([_packet(status="none")], NOW)
    assert ran == ["Puka Nacua"]


def test_a_quiet_note_does_not_start_the_agent():
    p = _packet(status="none")
    quiet = news.NewsItem(id="news.espn_01", url="u", published="p",
                          text="Chase brought in seven of nine targets for 75 yards (LA).")
    from dataclasses import replace
    assert not news._open_question(replace(p, news=(quiet,)))
