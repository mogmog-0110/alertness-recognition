"""ストレス側の新しい cue（呼吸・瞬き頻度）と、呼吸数の推定そのもののテスト。"""

from __future__ import annotations

import numpy as np
from _helpers import FakeHistory, make_observation

from alertness.classifier.cues.blink_rate import BlinkRateCue
from alertness.classifier.cues.respiration import RespirationCue
from alertness.contracts import Features
from alertness.features.rppg import estimate_respiration


def _breathing(rpm: float, seconds: float, fs: float = 30.0, drift: float = 0.0):
    """指定の呼吸数の正弦波。drift を入れると、ゆっくりした照明変化を混ぜられる。"""
    t = np.arange(0.0, seconds, 1.0 / fs)
    wave = np.sin(2.0 * np.pi * (rpm / 60.0) * t)
    return wave + drift * t


def test_respiration_rate_is_recovered_from_a_clean_wave():
    rpm, quality = estimate_respiration(_breathing(15.0, 40.0), 30.0)
    assert abs(rpm - 15.0) < 1.0
    assert quality > 0.0


def test_respiration_survives_a_slow_illumination_drift():
    # 直線の傾きを抜かないと、帯域の下端にできた電力が常に最強のピークになる。
    rpm, _ = estimate_respiration(_breathing(18.0, 40.0, drift=0.5), 30.0)
    assert abs(rpm - 18.0) < 1.5


def test_respiration_needs_enough_samples():
    rpm, quality = estimate_respiration(np.zeros(4), 30.0)
    assert np.isnan(rpm)
    assert quality == 0.0


def _resp_frames(rpm: float, seconds: float, t0: float = 0.0, quality: float = 0.7):
    n = int(seconds * 2)  # 0.5 秒刻み
    return [Features({"resp_rpm": rpm, "resp_quality": quality}, t0 + i * 0.5) for i in range(n)]


def _feed(cue, frames, step: int = 4):
    result = None
    for i in range(0, len(frames), step):
        result = cue.evaluate(make_observation(frames[i], FakeHistory(frames[: i + 1])))
    return result


def test_respiration_cue_is_silent_until_the_baseline_is_ready():
    # 安静基準ができる前は、呼吸が速い人を必ずストレス高と出してしまう。
    cue = RespirationCue(baseline_seconds=120.0)
    result = _feed(cue, _resp_frames(22.0, 20.0))
    assert result.score == 0.0
    assert not result.valid
    assert "測定中" in result.detail


def test_respiration_cue_detects_a_rise_over_resting():
    cue = RespirationCue(span_rpm=4.0, baseline_seconds=90.0, min_rest_samples=20)
    frames = _resp_frames(14.0, 150.0) + _resp_frames(20.0, 30.0, t0=150.0)
    result = _feed(cue, frames)
    assert result.valid
    assert result.score >= 0.8
    assert not result.active  # 単独ではストレスを断定しない


def test_respiration_cue_ignores_a_steadily_fast_breather():
    # ずっと同じ速さなら、本人基準に対して上がっていない＝ストレスではない。
    cue = RespirationCue(span_rpm=4.0, baseline_seconds=90.0, min_rest_samples=20)
    assert _feed(cue, _resp_frames(22.0, 180.0)).score < 0.3


def test_respiration_cue_stays_invalid_without_rppg():
    frames = [Features({"ear_norm": 1.0}, i * 0.5) for i in range(60)]
    result = _feed(cue=RespirationCue(), frames=frames)
    assert not result.valid
    assert result.score == 0.0


STEP = 0.05


def _blink_frames(per_minute: float, seconds: float, t0: float = 0.0):
    """指定頻度で 0.15 秒の瞬きが入る ear_norm 列。"""
    period = 60.0 / per_minute
    values: list[Features] = []
    n = int(seconds / STEP)
    for i in range(n):
        t = i * STEP
        phase = t % period
        ear = 0.2 if phase < 0.15 else 1.0
        values.append(Features({"ear_norm": ear, "yaw_rel": 0.0}, t0 + t))
    return values


def test_blink_rate_counts_blinks_per_minute():
    cue = BlinkRateCue(window_seconds=60.0)
    frames = _blink_frames(20.0, 60.0)
    rate = cue._rate(make_observation(frames[-1], FakeHistory(frames)))
    assert 17.0 <= rate <= 23.0


def test_blink_rate_excludes_long_closures():
    # 長い閉眼は瞬きではなく眠気。数に混ぜると、眠くなるほどストレスが上がって見える。
    cue = BlinkRateCue(window_seconds=60.0, max_blink_seconds=0.5)
    frames = _blink_frames(10.0, 60.0)
    frames += [Features({"ear_norm": 0.2, "yaw_rel": 0.0}, 60.0 + i * STEP) for i in range(40)]
    frames += [Features({"ear_norm": 1.0, "yaw_rel": 0.0}, 62.0 + i * STEP) for i in range(20)]
    rate = cue._rate(make_observation(frames[-1], FakeHistory(frames)))
    assert rate < 13.0  # 2秒の閉眼は1回に数えない


def test_blink_rate_is_silent_until_the_baseline_is_ready():
    cue = BlinkRateCue(baseline_seconds=600.0)
    frames = _blink_frames(25.0, 70.0)
    result = _feed(cue, frames, step=20)
    assert result.score == 0.0
    assert not result.valid
