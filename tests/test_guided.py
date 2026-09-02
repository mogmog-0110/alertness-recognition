"""ガイド付き収録の進行ロジックのテスト。"""

from __future__ import annotations

from alertness.guided import PROTOCOLS, GuidedSession, Prompt


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


def test_the_long_protocol_outlasts_every_cue_window():
    """1 状態の保持が、窓を使う cue の窓より長いこと。

    acted は 1 状態 12 秒だが eye_closure は 30 秒、blink_dynamics は 60 秒の窓で
    判定する。短い保持を並べると、どの時点の窓も複数のラベルを含んでしまい、
    これらの cue は原理的にラベルを分離できない。較正用の収録では、窓が単一
    ラベルで満たされる長さが要る。
    """
    longest_window_seconds = 60.0  # blink_dynamics
    for prompt in PROTOCOLS["acted_long"]:
        assert prompt.hold_seconds > longest_window_seconds, prompt.label


def test_the_long_protocol_covers_the_same_states():
    # 較正のためだけに状態を減らさない。短い版と同じラベルを揃える。
    assert [p.label for p in PROTOCOLS["acted_long"]] == [
        p.label for p in PROTOCOLS["acted"]
    ]
