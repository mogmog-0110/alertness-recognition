"""EAR（目の縦横比）の計算。"""

from __future__ import annotations

from collections.abc import Sequence

from ..geometry import Point, euclidean


def eye_aspect_ratio(points: Sequence[Point]) -> float:
    """片目6点から EAR を求める。

    points は (p1, p2, p3, p4, p5, p6)。
    縦2方向の距離平均を横幅で割る。小さいほど目が閉じている。
    """
    if len(points) != 6:
        raise ValueError("EAR には6点が必要です")
    horizontal = euclidean(points[0], points[3])
    if horizontal <= 1e-6:
        return 0.0
    vertical = euclidean(points[1], points[5]) + euclidean(points[2], points[4])
    return vertical / (2.0 * horizontal)


def is_eye_closed(ear: float, open_baseline: float, closed_ratio: float) -> bool:
    """開眼基準に対する割合で閉眼を判定する。

    例: closed_ratio=0.6 なら、基準の6割未満まで下がったら閉眼とみなす。
    基準値を使うことで、目の形の個人差を吸収する。
    """
    if open_baseline <= 1e-6:
        return False
    return ear < open_baseline * closed_ratio
