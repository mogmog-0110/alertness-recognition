from __future__ import annotations

from alertness.contracts import CalibrationProfile
from alertness.ingest.autocalibrate import estimate_profile


def _row(ear: float) -> dict:
    return {
        "ear": ear,
        "mar": 0.0,
        "pitch": 10.0,
        "yaw": 5.0,
        "roll": 0.0,
        "gaze_x": 0.5,
        "face_scale": 100.0,
    }


def test_open_baseline_uses_upper_percentile():
    # 実データと同様、開眼(0.35)が多数で閉眼(0.1)は少数。
    rows = [_row(0.35) for _ in range(7)] + [_row(0.1) for _ in range(3)]
    profile = estimate_profile(rows, "s1")

    assert profile.user_id == "s1"
    # 上位側を採るので、開眼基準は閉眼値(0.1)ではなく開眼側に寄る。
    assert profile.ear_open_baseline > 0.2
    assert abs(profile.head_pose_neutral.pitch - 10.0) < 1e-6


def test_empty_rows_return_identity():
    profile = estimate_profile([], "x")
    assert isinstance(profile, CalibrationProfile)
