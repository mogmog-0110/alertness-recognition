"""統計的キャリブレーションと、遅れて出る特徴（心拍）の安静基準取得。"""

from __future__ import annotations

from _helpers import make_observation

from alertness.calibration.calibrator import StatisticalCalibrator
from alertness.contracts import Features


def _feed(cal: StatisticalCalibrator, values: dict, times) -> None:
    for t in times:
        cal.collect(make_observation(Features(values=dict(values), timestamp=t)))


def test_geometric_calibration_finishes_in_a_few_seconds():
    cal = StatisticalCalibrator(duration_seconds=1.0, fps=30.0, warmup_seconds=0.0)
    _feed(cal, {"ear": 0.3, "gaze_y": 0.5}, [i / 30.0 for i in range(30)])

    assert cal.progress >= 1.0
    profile = cal.finalize()
    assert profile.baselines["ear"] == 0.3
    assert profile.baselines["gaze_y"] == 0.5


def test_warmup_frames_are_excluded_from_baseline():
    cal = StatisticalCalibrator(duration_seconds=1.0, fps=30.0, warmup_seconds=0.5)
    # 前半（ウォームアップ）は極端な値、後半が本当の中立。
    for i in range(15):
        cal.collect(make_observation(Features(values={"ear": 99.0}, timestamp=i / 30.0)))
    for i in range(15, 45):
        cal.collect(make_observation(Features(values={"ear": 0.3}, timestamp=i / 30.0)))

    assert cal.finalize().baselines["ear"] == 0.3  # ウォームアップの99は捨てられる


def test_calibration_waits_for_delayed_heart_rate():
    # rPPG は最初の数秒は出ない。心拍が必要なら、出るまで完了しない。
    cal = StatisticalCalibrator(
        duration_seconds=1.0, fps=30.0, warmup_seconds=0.0,
        require_keys=("hr_bpm",), max_seconds=5.0,
    )
    # 幾何は満ちるが心拍はまだ無い → 完了しない。
    _feed(cal, {"ear": 0.3}, [i / 30.0 for i in range(30)])
    assert cal.progress < 1.0

    # 心拍が出始めると完了へ向かう。
    for i in range(30, 60):
        feats = Features(values={"ear": 0.3, "hr_bpm": 72.0}, timestamp=i / 30.0)
        cal.collect(make_observation(feats))
    assert cal.progress >= 1.0
    assert cal.finalize().baselines["hr_bpm"] == 72.0


def test_calibration_gives_up_waiting_at_the_cap():
    # 心拍が一度も出なくても、上限に達したら確定する（心拍なしカメラでも止まらない）。
    cal = StatisticalCalibrator(
        duration_seconds=1.0, fps=30.0, warmup_seconds=0.0,
        require_keys=("hr_bpm",), max_seconds=2.0,
    )
    _feed(cal, {"ear": 0.3}, [i / 30.0 for i in range(60)])  # 2秒ぶん、心拍なし

    assert cal.progress >= 1.0
    baselines = cal.finalize().baselines
    assert "ear" in baselines
    assert "hr_bpm" not in baselines  # 出なかったので基準も無い（推論側は present で扱う）
