"""前方基準の自動推定のテスト。

起動時キャリブのずれを、走行中の分布から取り戻せるかを見る。
"""

from __future__ import annotations

from _helpers import FakeHistory, make_observation

from alertness.classifier.cues._forward import ForwardBaseline, ForwardPose
from alertness.classifier.cues.attention_buffer import AttentionBufferCue
from alertness.contracts import Features


def _feed(baseline: ForwardBaseline, values, step: float = 1.0):
    for i, value in enumerate(values):
        baseline.update(i * step, value)
    return baseline


def test_the_mode_is_found_not_the_mean():
    # 8割は 12 度（＝取り付けのずれ）、2割は 40 度（ミラー確認）。
    # 平均も中央値もミラー側へ引かれるが、最頻値は「一番長く留まった向き」を残す。
    values = [12.0] * 80 + [40.0] * 20
    center, ready = _feed(ForwardBaseline(bin_width=2.0, min_samples=50), values).read()
    assert ready
    assert abs(center - 12.0) < 2.0


def test_a_flat_distribution_is_not_confident():
    # たまたま多かっただけの向きを前方に据えないこと。
    values = [float(i % 60) for i in range(120)]
    _, ready = _feed(ForwardBaseline(bin_width=2.0, min_samples=50), values).read()
    assert not ready


def test_too_few_samples_are_not_confident():
    _, ready = _feed(ForwardBaseline(bin_width=2.0, min_samples=50), [10.0] * 10).read()
    assert not ready


def test_samples_are_spaced_in_time():
    # 毎フレーム積むと直近の数秒が分布を乗っ取る。
    baseline = ForwardBaseline(bin_width=2.0, interval=1.0)
    for i in range(600):
        baseline.update(i * 0.1, 10.0)  # 60 秒ぶんを 0.1 秒刻みで
    assert len(baseline._samples) <= 65


def test_old_samples_leave_the_window():
    baseline = ForwardBaseline(bin_width=2.0, seconds=10.0, interval=1.0)
    for i in range(60):
        baseline.update(float(i), 10.0)
    assert all(t >= 49.0 for t, _ in baseline._samples)


def test_correction_subtracts_the_estimated_forward():
    pose = ForwardPose(min_samples=30, interval=0.5)
    for i in range(120):
        pose.update(i * 0.5, 0.05, 12.0, 4.0)
    assert pose.ready
    gaze, yaw, pitch = pose.correct(0.05, 12.0, 4.0)
    assert abs(gaze) < 0.005
    assert abs(yaw) < 1.0
    assert abs(pitch) < 1.0


def test_an_unconfident_axis_passes_the_value_through():
    pose = ForwardPose(min_samples=1000)
    assert pose.correct(0.2, 30.0, 5.0) == (0.2, 30.0, 5.0)


def test_a_miscalibrated_mounting_stops_draining_the_buffer():
    # カメラが 30 度ずれて付いている状況。前方を見続けているのに yaw_rel が 30 度出る
    # （on_target_yaw_deg の 25 度を超えるので、いまの実装では脇見と読まれる）。
    # 自動推定が無いと、注意残高が減り続けて誤警告になる。
    def run(auto: bool) -> float:
        cue = AttentionBufferCue(capacity_seconds=2.0, auto_forward=auto)
        frames = [
            Features({"gaze_dx": 0.0, "gaze_off": 0.0, "yaw_rel": 30.0, "pitch_rel": 0.0}, i * 0.5)
            for i in range(300)
        ]
        result = None
        for i in range(len(frames)):
            result = cue.evaluate(make_observation(frames[i], FakeHistory(frames[: i + 1])))
        return result.score

    assert run(auto=False) == 0.0  # ずれたまま＝残高が尽きる（いまの振る舞い）
    assert run(auto=True) >= 1.0  # 分布から前方を学び直して回復する
