from __future__ import annotations

import math
from collections.abc import Sequence

DEFAULT_WEIGHTS = {
    "theta": 0.35,
    "alpha": 0.25,
    "beta": -0.20,
    "di": 0.25,
    "sem": 0.20,
    "blink_duration": 0.15,
    "microsleep_duration": 0.20,
}


def compute_cds(
    features: Sequence[dict[str, float]],
    *,
    weights: dict[str, float] | None = None,
    sigmoid_center: float = 0.0,
    sigmoid_scale: float = 1.0,
) -> list[float]:
    """符号付き特徴量を統合し、シグモイドで0〜100のCDSへ写す。"""
    if sigmoid_scale <= 0:
        raise ValueError("sigmoid_scale は正の値である必要があります")
    active_weights = weights or DEFAULT_WEIGHTS
    scores: list[float] = []
    for item in features:
        raw = sum(
            float(weight) * float(item.get(name, 0.0)) for name, weight in active_weights.items()
        )
        exponent = max(-60.0, min(60.0, -(raw - sigmoid_center) / sigmoid_scale))
        score = 100.0 / (1.0 + math.exp(exponent))
        scores.append(float(score))
    return scores
