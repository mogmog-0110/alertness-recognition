from __future__ import annotations

from collections.abc import Sequence


def calibrate_thresholds(
    thresholds: Sequence[float],
    impairment: float,
    *,
    gain: float = 5.0,
    max_shift: float = 10.0,
) -> tuple[float, float, float]:
    """PVT1から悪化したセッションではLoD境界を設定範囲内で下げる。"""
    if len(thresholds) != 3:
        raise ValueError("thresholds は3値である必要があります")
    shift = max(-max_shift, min(max_shift, float(impairment) * gain))
    return tuple(float(value) - shift for value in thresholds)  # type: ignore[return-value]


def classify_lod(
    scores: Sequence[float], *, thresholds: Sequence[float] = (20.0, 50.0, 75.0)
) -> list[str]:
    """CDS を None / Low / Medium / High に変換する。"""
    if len(thresholds) != 3 or list(thresholds) != sorted(thresholds):
        raise ValueError("thresholds は昇順の3値である必要があります")
    low, medium, high = (float(value) for value in thresholds)
    levels: list[str] = []
    for score in scores:
        if score < low:
            levels.append("none")
        elif score < medium:
            levels.append("low")
        elif score < high:
            levels.append("medium")
        else:
            levels.append("high")
    return levels
