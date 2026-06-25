"""座標計算の共通ヘルパ。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Point = Sequence[float]


def euclidean(p: Point, q: Point) -> float:
    # 2点間のユークリッド距離。
    a = np.asarray(p, dtype=float)
    b = np.asarray(q, dtype=float)
    return float(np.linalg.norm(a - b))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    # 値を [low, high] に収める。
    return max(low, min(high, value))
