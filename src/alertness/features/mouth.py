"""MAR（口の縦横比）の計算。あくび検出に使う。"""

from __future__ import annotations

from ..geometry import Point, euclidean


def mouth_aspect_ratio(top: Point, bottom: Point, left: Point, right: Point) -> float:
    """口の開きを口幅で割った比。大きいほど口が開いている。

    top/bottom は上下の唇、left/right は口角。
    """
    width = euclidean(left, right)
    if width <= 1e-6:
        return 0.0
    return euclidean(top, bottom) / width
