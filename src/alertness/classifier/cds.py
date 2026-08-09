"""基準化済みPSG特徴を、0〜100の連続眠気スコア（CDS）へ統合する。

EEGの相対帯域パワーと眠気指標（theta、alpha、DI）、覚醒方向の beta、EOG由来のSEM・
瞬目時間・長時間閉眼相当時間を符号付き重みで線形結合し、シグモイドで共通スケールへ写す。
ここでは連続値の生成だけを担当し、none/low/medium/high の段階化や時間平滑化は行わない。

主な呼び出し元は ``examples/convert_drozy.py`` で、被験者内基準化後の特徴を受け取る。実運用の
重みとシグモイド設定は ``config/default.yaml`` の ``drozy.cds`` が正準で、ここにある既定値は
設定が省略された場合と単体利用のためのフォールバックである。
"""

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
