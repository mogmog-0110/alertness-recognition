"""PSG眠気特徴の被験者内基準化と、補助的なスコア校正を担う。

DROZYは被験者間で信号振幅や特徴量の水準が異なるため、``fit_baseline`` が PVT1 冒頭の有効な
PSG窓から各特徴の平均・標準偏差を求め、``normalize_features`` が後続セッションを同じ
z-score空間へ写す。ゼロ分散の特徴は除外し、十分な基準標本がない被験者は変換側でスキップ
できるよう例外を返す。この統計は ``examples/convert_drozy.py`` から CDS 計算の直前に使われる。

``calibrate_with_pvt_kss`` と ``normalize_feature_series`` は、配列ベースの簡易パイプラインとの
互換を保つ補助関数である。正式なDROZY manifest経路では、PVT/KSSをここでCDSへ加算せず、
PVTは ``classifier.lod`` の境界補正、KSSは変換結果の検証に限定する。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

FEATURE_NAMES = (
    "theta",
    "alpha",
    "beta",
    "di",
    "sem",
    "blink_duration",
    "microsleep_duration",
)


@dataclass(frozen=True)
class BaselineStats:
    means: dict[str, float]
    stds: dict[str, float]
    valid_features: tuple[str, ...]
    sample_count: int


def _values(item: Any) -> dict[str, float]:
    if isinstance(item, dict):
        return {name: float(item.get(name, float("nan"))) for name in FEATURE_NAMES}
    if hasattr(item, "values"):
        values = item.values()
        return {name: float(values.get(name, float("nan"))) for name in FEATURE_NAMES}
    return {name: float(getattr(item, name, float("nan"))) for name in FEATURE_NAMES}


def fit_baseline(
    features: Sequence[Any],
    *,
    baseline_seconds: float = 120.0,
    min_std: float = 1e-6,
) -> BaselineStats:
    """PVT1冒頭の有効特徴量から被験者別の平均・標準偏差を求める。"""
    selected: list[dict[str, float]] = []
    for index, item in enumerate(features):
        timestamp = float(getattr(item, "timestamp", index))
        valid = bool(getattr(item, "valid", True))
        if timestamp > baseline_seconds:
            break
        values = _values(item)
        if valid and all(np.isfinite(list(values.values()))):
            selected.append(values)
    if len(selected) < 2:
        raise ValueError("PVT1基準化に必要な有効特徴量が2件以上ありません")
    means = {name: float(np.mean([row[name] for row in selected])) for name in FEATURE_NAMES}
    stds = {name: float(np.std([row[name] for row in selected])) for name in FEATURE_NAMES}
    valid_features = tuple(name for name in FEATURE_NAMES if stds[name] >= min_std)
    if not valid_features:
        raise ValueError("PVT1基準区間の全特徴量がゼロ分散です")
    return BaselineStats(means, stds, valid_features, len(selected))


def normalize_features(features: Sequence[Any], baseline: BaselineStats) -> list[dict[str, float]]:
    """基準統計が有効な特徴だけをz-scoreへ変換する。"""
    output: list[dict[str, float]] = []
    for item in features:
        values = _values(item)
        row = {
            name: (values[name] - baseline.means[name]) / baseline.stds[name]
            for name in baseline.valid_features
        }
        output.append(row)
    return output


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
            pvt_value = pvt_values[min(index, len(pvt_values) - 1)]
            pvt_scale = 1.0 + min(0.25, max(0.0, pvt_value / 1000.0))
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
