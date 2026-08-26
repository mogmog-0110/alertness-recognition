"""視線の水平比の推定。虹彩中心が目の幅のどこにあるかで左右を見る。"""

from __future__ import annotations

from ..contracts import FaceLandmarks
from ..geometry import clamp
from . import landmark_ids as ids


def horizontal_gaze_ratio(landmarks: FaceLandmarks) -> float:
    """0=画像の左端寄り, 1=画像の右端寄り, 0.5=目の中央。両目の平均を返す。

    基準を「目頭側／目尻側」ではなく画像の左右に取る。目頭と目尻の並びは左右の目で
    鏡像なので、目頭を 0 と決めると、左右を見たとき両目の比が逆向きに出て、平均が
    打ち消し合う（＝どちらを向いても 0.5 のまま動かない）。

    虹彩点を持たない（478点未満の）モデルでは 0.5 を返す。
    """
    if landmarks.points.shape[0] <= ids.RIGHT_IRIS:
        return 0.5
    left = _eye_ratio(landmarks, ids.LEFT_IRIS, ids.LEFT_EYE_INNER, ids.LEFT_EYE_OUTER)
    right = _eye_ratio(landmarks, ids.RIGHT_IRIS, ids.RIGHT_EYE_INNER, ids.RIGHT_EYE_OUTER)
    return (left + right) / 2.0


def _eye_ratio(lm: FaceLandmarks, iris_id: int, inner_id: int, outer_id: int) -> float:
    """片目の虹彩位置。目の中央からのズレを目の幅で割り、0.5 を中央に置く。

    幅は絶対値で取る。目頭と目尻のどちらが画像の左に来るかは左右の目で入れ替わるが、
    ここで欲しいのは「画像上でどちらへ寄っているか」なので、並びの向きに依らせない。
    """
    iris = lm.pixel(iris_id)
    inner = lm.pixel(inner_id)
    outer = lm.pixel(outer_id)
    span = abs(outer[0] - inner[0])
    if span < 1e-6:
        return 0.5
    center = (inner[0] + outer[0]) / 2.0
    return clamp(0.5 + (iris[0] - center) / span)
