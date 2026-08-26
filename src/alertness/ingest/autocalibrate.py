"""外部動画から中立姿勢・開眼基準を推定する自動キャリブレーション。

アプリのキャリブは「本人が中立姿勢で数秒待つ」前提だが、公開データにその工程は無い。
そこで動画全体の生特徴量の分布から、中立姿勢＝各角度の中央値、開眼基準＝EARの上位値、
のように統計的に推定する。姿勢が動く動画でも大きく外さない狙い。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..contracts import CalibrationProfile, Pose

# 開眼基準は「楽に開けたとき」を狙うので、中央値ではなく上位側の値を採る。
_EAR_OPEN_PERCENTILE = 85.0


def _percentile(values: Sequence[float], q: float, default: float) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return default
    return float(np.percentile(arr, q))


def _column(rows: Sequence[Mapping[str, float]], key: str) -> list[float]:
    return [float(r[key]) for r in rows if key in r]


def estimate_profile(
    rows: Sequence[Mapping[str, float]], subject: str = "default"
) -> CalibrationProfile:
    """生特徴量（ear/mar/pitch/yaw/roll/gaze_x/face_scale）の並びから基準を作る。

    顔が映っているフレームの値だけを渡す前提。空なら恒等プロファイルを返す。
    """
    if not rows:
        return CalibrationProfile.identity()

    ear_open = _percentile(_column(rows, "ear"), _EAR_OPEN_PERCENTILE, 0.3)
    mar_neutral = _percentile(_column(rows, "mar"), 50.0, 0.0)
    pitch = _percentile(_column(rows, "pitch"), 50.0, 0.0)
    yaw = _percentile(_column(rows, "yaw"), 50.0, 0.0)
    roll = _percentile(_column(rows, "roll"), 50.0, 0.0)
    gaze_x = _percentile(_column(rows, "gaze_x"), 50.0, 0.5)
    gaze_y = _percentile(_column(rows, "gaze_y"), 50.0, 0.5)
    face_scale = _percentile(_column(rows, "face_scale"), 50.0, 1.0)

    return CalibrationProfile(
        ear_open_baseline=ear_open,
        mar_neutral=mar_neutral,
        head_pose_neutral=Pose(pitch=pitch, yaw=yaw, roll=roll),
        gaze_center=(gaze_x, gaze_y),
        face_scale=face_scale,
        user_id=subject or "default",
    )
