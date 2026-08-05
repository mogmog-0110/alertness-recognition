from __future__ import annotations

from collections.abc import Sequence


def compute_cds(features: Sequence[dict[str, float]]) -> list[float]:
    """単純な線形統合で CDS を算出する。"""
    scores: list[float] = []
    for item in features:
        theta = item.get("theta", 0.0)
        alpha = item.get("alpha", 0.0)
        beta = item.get("beta", 0.0)
        di = item.get("di", 0.0)
        sem = item.get("sem", 0.0)
        blink_duration = item.get("blink_duration", 0.0)
        microsleep_duration = item.get("microsleep_duration", 0.0)

        raw = 0.35 * theta + 0.25 * alpha + 0.25 * di + 0.20 * sem + 0.15 * blink_duration + 0.20 * microsleep_duration - 0.20 * beta
        score = max(0.0, min(100.0, raw * 10.0))
        scores.append(float(score))
    return scores
