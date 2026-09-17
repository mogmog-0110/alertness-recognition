"""cue 共通の時系列ヘルパ（時間で数える割合と被覆率）のテスト。"""

from __future__ import annotations

import pytest
from _helpers import FakeHistory, make_observation

from alertness.classifier.cues._support import sample_durations, time_fraction, window_coverage
from alertness.classifier.cues.eye_closure import EyeClosureCue
from alertness.contracts import Features


def _fps_drop_frames(closed_in_slow_part: bool):
    """30fps で 5 秒、そのあと 5fps で 25 秒。遅い側だけ目を閉じた区間を混ぜられる。"""
    frames = [Features({"ear_norm": 1.0}, i / 30.0) for i in range(150)]
    for i in range(125):
        closed = closed_in_slow_part and i < 50  # 遅い側の 10 秒＝窓の 1/3
        frames.append(Features({"ear_norm": 0.2 if closed else 1.0}, 5.0 + i * 0.2))
    return frames


def test_coverage_stays_full_when_fps_drops_mid_window():
    # 中央値の倍数で頭打ちにすると、遅い側の 0.2 秒間隔が 0.067 秒に削られて 0.45 になる。
    frames = _fps_drop_frames(closed_in_slow_part=False)
    obs = make_observation(frames[-1], FakeHistory(frames))
    assert window_coverage(obs, 30.0) > 0.95


def test_perclos_counts_time_not_frames_after_an_fps_drop():
    frames = _fps_drop_frames(closed_in_slow_part=True)
    obs = make_observation(frames[-1], FakeHistory(frames))
    result = EyeClosureCue(window_seconds=30.0, perclos_drowsy=0.4).evaluate(obs)
    assert result.valid
    times = [f.timestamp for f in frames]
    perclos = time_fraction(times, [f.get("ear_norm") < 0.6 for f in frames])
    assert perclos == pytest.approx(1.0 / 3.0, abs=0.02)


def test_a_long_tracking_loss_is_not_credited_to_the_previous_sample():
    times = [0.0, 0.1, 0.2, 10.2, 10.3]
    durations = sample_durations(times)
    assert durations[2] == pytest.approx(0.5)  # 10 秒の空白は上限まで
    # 頭打ちが無ければ 10/10.3 になり、見えていなかった時間がほぼ全部 True に数えられる。
    assert time_fraction(times, [False, False, True, False, False]) == pytest.approx(0.5 / 0.9)


def test_the_gap_cap_is_a_parameter():
    times = [0.0, 1.0, 2.0]
    assert sample_durations(times, max_gap=2.0) == [1.0, 1.0, 1.0]
    assert sample_durations(times, max_gap=0.0) == [1.0, 1.0, 1.0]  # 0 以下は頭打ちなし
    assert sample_durations(times, max_gap=0.25) == [0.25, 0.25, 0.25]
