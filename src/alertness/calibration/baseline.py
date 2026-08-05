from __future__ import annotations

from collections.abc import Sequence


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
