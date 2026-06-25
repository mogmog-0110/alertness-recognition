"""視線の水平比の推定。虹彩中心が目頭・目尻のどこにあるかで左右を見る。"""

from __future__ import annotations

from ..contracts import FaceLandmarks
from ..geometry import clamp
from . import landmark_ids as ids


def horizontal_gaze_ratio(landmarks: FaceLandmarks) -> float:
    """0=目頭側, 1=目尻側, 0.5=中央。両目の平均を返す。

    虹彩点を持たない（478点未満の）モデルでは 0.5 を返す。
    """
    if landmarks.points.shape[0] <= ids.RIGHT_IRIS:
        return 0.5
    left = _eye_ratio(landmarks, ids.LEFT_IRIS, ids.LEFT_EYE_INNER, ids.LEFT_EYE_OUTER)
    right = _eye_ratio(landmarks, ids.RIGHT_IRIS, ids.RIGHT_EYE_INNER, ids.RIGHT_EYE_OUTER)
    return (left + right) / 2.0


def _eye_ratio(lm: FaceLandmarks, iris_id: int, inner_id: int, outer_id: int) -> float:
    iris = lm.pixel(iris_id)
    inner = lm.pixel(inner_id)
    outer = lm.pixel(outer_id)
    span = outer[0] - inner[0]
    if abs(span) < 1e-6:
        return 0.5
    return clamp((iris[0] - inner[0]) / span)
