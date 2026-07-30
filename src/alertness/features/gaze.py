"""視線比の推定。虹彩中心が目のどこにあるかで、見ている向きを見る。"""

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
    left = _horizontal(landmarks, ids.LEFT_IRIS, ids.LEFT_EYE_INNER, ids.LEFT_EYE_OUTER)
    right = _horizontal(landmarks, ids.RIGHT_IRIS, ids.RIGHT_EYE_INNER, ids.RIGHT_EYE_OUTER)
    return (left + right) / 2.0


def vertical_gaze_ratio(landmarks: FaceLandmarks) -> float:
    """0=上, 1=下, 0.5=中央。両目の平均を返す。

    水平比は目頭・目尻という動かない2点の間で位置を測れるが、縦にはそれが無い。
    上下まぶたは瞬きと開度で動くので基準に使えない。そこで目頭と目尻を結ぶ線を基準線とし、
    虹彩がそこからどれだけ上下にずれているかを、目の幅で割って尺度をそろえる。
    幅で割るので顔の遠近には依存しない。
    """
    if landmarks.points.shape[0] <= ids.RIGHT_IRIS:
        return 0.5
    left = _vertical(landmarks, ids.LEFT_IRIS, ids.LEFT_EYE_INNER, ids.LEFT_EYE_OUTER)
    right = _vertical(landmarks, ids.RIGHT_IRIS, ids.RIGHT_EYE_INNER, ids.RIGHT_EYE_OUTER)
    return (left + right) / 2.0


def _horizontal(lm: FaceLandmarks, iris_id: int, inner_id: int, outer_id: int) -> float:
    iris = lm.pixel(iris_id)
    inner = lm.pixel(inner_id)
    outer = lm.pixel(outer_id)
    span = outer[0] - inner[0]
    if abs(span) < 1e-6:
        return 0.5
    return clamp((iris[0] - inner[0]) / span)


def _vertical(lm: FaceLandmarks, iris_id: int, inner_id: int, outer_id: int) -> float:
    iris = lm.pixel(iris_id)
    inner = lm.pixel(inner_id)
    outer = lm.pixel(outer_id)
    span = abs(outer[0] - inner[0])
    if span < 1e-6:
        return 0.5
    baseline = (inner[1] + outer[1]) / 2.0  # 目頭と目尻を結ぶ線の高さ
    return clamp(0.5 + (iris[1] - baseline) / span)
