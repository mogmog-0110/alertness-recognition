"""サングラス・暗所での縮退運転のテスト。

顔検出は成功するが目の信号が使えない、という状態を作って、
(1) 目に依存する cue が黙ること、(2) 残る cue で判定が続くこと、を確かめる。
"""

from __future__ import annotations

from _helpers import FakeHistory, make_observation

from alertness.classifier.cues._eye_health import eye_signal_usable
from alertness.classifier.cues.blink import BlinkCue
from alertness.classifier.cues.eye_closure import EyeClosureCue
from alertness.classifier.policies.rule_based import RuleBasedPolicy
from alertness.classifier.states import DimensionSpec
from alertness.contracts import CueResult, Features, Level

STEP = 0.05
LEVELS = {"low": 0.3, "medium": 0.6, "high": 0.8}


def _stuck_low(seconds: float = 90.0):
    """サングラス越しの EAR。低い値のまま動かない（瞬きが出ない）。"""
    n = int(seconds / STEP)
    return [Features({"ear_norm": 0.45, "yaw_rel": 0.0}, i * STEP) for i in range(n)]


def _blinking(seconds: float = 90.0, period: float = 4.0):
    n = int(seconds / STEP)
    frames = []
    for i in range(n):
        t = i * STEP
        ear = 0.2 if (t % period) < 0.15 else 1.0
        frames.append(Features({"ear_norm": ear, "yaw_rel": 0.0}, t))
    return frames


def _obs(frames):
    return make_observation(frames[-1], FakeHistory(frames))


def test_a_signal_with_blinks_is_usable():
    usable, reason = eye_signal_usable(_obs(_blinking()))
    assert usable
    assert reason == ""


def test_a_signal_without_any_blink_is_rejected():
    # 人は安静時でも 10〜20 回/分は瞬きする。1分間に1回も無いのは目が見えていない証拠。
    usable, reason = eye_signal_usable(_obs(_stuck_low()))
    assert not usable
    assert "瞬き" in reason


def test_a_short_history_is_not_mistaken_for_sunglasses():
    # 起動直後をサングラスと読むと、最も判定が要る立ち上がりが丸ごと縮退運転になる。
    usable, _ = eye_signal_usable(_obs(_stuck_low(seconds=5.0)))
    assert usable


def test_eye_closure_does_not_cry_drowsy_through_sunglasses():
    # EAR が低いまま張り付くと PERCLOS が満点になり、目を閉じていないのに眠気を警告する。
    # 最も避けたい向きの誤りなので、黙って頭部の cue に譲る。
    result = EyeClosureCue().evaluate(_obs(_stuck_low()))
    assert not result.active
    assert result.score == 0.0
    assert not result.valid


def test_blink_does_not_report_a_permanent_microsleep():
    result = BlinkCue().evaluate(_obs(_stuck_low()))
    assert not result.active
    assert not result.valid


def test_eye_cues_still_work_when_blinks_are_present():
    assert EyeClosureCue().evaluate(_obs(_blinking())).valid
    assert BlinkCue().evaluate(_obs(_blinking())).valid


def test_the_remaining_cues_still_drive_the_axis():
    # 目の cue が4本落ちても、残る頭部の cue が満点なら警告に届くこと。
    # 分母を全 cue に固定していると 3/7 にしかならず、縮退運転が成立しない。
    spec = DimensionSpec(
        "drowsiness",
        LEVELS,
        ("eye_closure", "blink", "blink_dynamics", "blink_rate", "nodding", "head_down", "yawn"),
    )
    policy = RuleBasedPolicy([spec], dict.fromkeys(spec.cues, 1.0), 1, 1)
    dead = [
        CueResult(name, "drowsiness", 0.0, False, "", None, False)
        for name in ("eye_closure", "blink", "blink_dynamics", "blink_rate")
    ]
    alive = [
        CueResult(name, "drowsiness", 0.9, False, "", None, True)
        for name in ("nodding", "head_down", "yawn")
    ]
    result = policy.decide(make_observation(Features({}, 0.0)), [*dead, *alive])
    assert result.dimensions["drowsiness"].level >= Level.HIGH
