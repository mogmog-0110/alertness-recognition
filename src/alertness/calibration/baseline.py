from __future__ import annotations

from collections.abc import Sequence


def calibrate_with_pvt_kss(
    scores: Sequence[float],
    *,
    pvt: Sequence[float] | None = None,
    kss: Sequence[float] | None = None,
) -> list[float]:
    """PVT/KSS を弱い補正として反映した CDS を返す。"""
    if not scores:
        return []

    calibrated: list[float] = []
    pvt_values = list(pvt or [])
    kss_values = list(kss or [])
    for index, score in enumerate(scores):
        adjusted = float(score)
        if pvt_values:
            pvt_scale = 1.0 + min(0.25, max(0.0, pvt_values[min(index, len(pvt_values) - 1)] / 1000.0))
            adjusted *= pvt_scale
        if kss_values:
            kss_bias = (kss_values[min(index, len(kss_values) - 1)] - 5.0) * 2.5
            adjusted += kss_bias
        calibrated.append(max(0.0, min(100.0, adjusted)))
    return calibrated


def normalize_feature_series(features: Sequence[dict[str, float]]) -> list[dict[str, float]]:
    """簡易の z-score 正規化。初期覚醒区間を基準にして個人差を吸収する。"""
    if not features:
        return []

    baseline = features[0]
    normalized: list[dict[str, float]] = []
    for item in features:
        row: dict[str, float] = {}
        for key in ("theta", "alpha", "beta", "di", "sem", "blink_duration", "microsleep_duration"):
            base_value = baseline.get(key, 0.0)
            current_value = item.get(key, 0.0)
            if key == "beta":
                row[key] = base_value - current_value
            else:
                row[key] = current_value - base_value
        normalized.append(row)
    return normalized
