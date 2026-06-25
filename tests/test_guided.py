"""ガイド付き収録の進行ロジックのテスト。"""

from __future__ import annotations

from alertness.guided import GuidedSession, Prompt


def _session():
    prompts = [
        Prompt("awake", "覚醒", "...", hold_seconds=10.0, ready_seconds=2.0),
        Prompt("drowsiness", "眠い", "...", hold_seconds=10.0, ready_seconds=2.0),
    ]
    return GuidedSession(prompts, rounds=1)


def test_starts_in_ready_without_label():
    s = _session()
    step = s.step(0.0)  # 最初の呼び出しで開始時刻を固定
    assert step.phase == "ready"
    assert step.label == ""


def test_hold_assigns_label():
    s = _session()
    s.step(0.0)
    step = s.step(5.0)  # ready(0-2) を過ぎて hold(2-12)
    assert step.phase == "hold"
    assert step.label == "awake"


def test_second_prompt_label():
    s = _session()
    s.step(0.0)
    step = s.step(17.0)  # 2つ目: ready(12-14)→hold(14-24)
    assert step.phase == "hold"
    assert step.label == "drowsiness"


def test_finishes_after_total():
    s = _session()
    s.step(0.0)
    step = s.step(100.0)  # 合計 24 秒を超える
    assert step.phase == "done"
    assert step.progress == 1.0
