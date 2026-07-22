"""学習済みモデル(model.pkl)を読んで判定する Classifier 実装。

ルールベースの CueClassifier の隣に置く、差し替え可能なもう一つの判定器。
cue は使わず、features を「学習時に保存した列順」でベクトル化し、軸ごとの
モデルで段階を予測する。学習(alertness-colab)が書き出した bundle
{models, features, classes} をそのまま受け取る。学習と推論で同じ列順・同じ
軸名になるのが唯一の取り決め。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import Assessment, Dimension, Level, Observation
from .states import DimensionSpec, alarm_of, level_for

# 学習のターゲット列 "label_<軸>" と、本体の評価軸名 "<軸>" をつなぐ接頭辞。
_AXIS_PREFIX = "label_"

_LEVEL_BY_NAME = {
    "none": Level.NONE,
    "low": Level.LOW,
    "medium": Level.MEDIUM,
    "high": Level.HIGH,
}


def _dimension_name(target: str) -> str:
    # "label_drowsiness" → "drowsiness"
    return target[len(_AXIS_PREFIX) :] if target.startswith(_AXIS_PREFIX) else target


def _level_of(name: str) -> Level:
    try:
        return _LEVEL_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(
            f"未知のレベル '{name}'。none/low/medium/high のいずれかであること。"
        ) from exc


class MLClassifier:
    """bundle の軸ごとのモデルで assess する。Classifier ポートの実装。"""

    # bundle は pickle 由来で中身の型が揃わないため Any で受ける。
    def __init__(
        self, bundle: Mapping[str, Any], dimensions: Sequence[DimensionSpec] | None = None
    ) -> None:
        models = bundle.get("models")
        features = bundle.get("features")
        if not models:
            raise ValueError("model.pkl に軸ごとのモデル(models)が入っていません。")
        if not features:
            raise ValueError("model.pkl に特徴量の列順(features)が入っていません。")
        self._models = dict(models)
        self._features = list(features)
        # 軸の向き（高いほど良い軸か）は config 側の取り決めなので、rule と同じ spec を使う。
        self._specs = {s.name: s for s in (dimensions or ())}

    def assess(self, obs: Observation) -> Assessment:
        # 欠損は 0.0（学習側の fillna(0.0) と揃える）。列順は bundle に従う。
        vector = [obs.features.get(name, 0.0) for name in self._features]
        dims: dict[str, Dimension] = {}
        for target, model in self._models.items():
            name = _dimension_name(target)
            level, score = self._predict(model, vector)
            dims[name] = self._as_dimension(name, score, level)
        return Assessment(dimensions=dims, timestamp=obs.features.timestamp)

    def _as_dimension(self, name: str, score: float, level: Level) -> Dimension:
        # 反転する軸（集中など）は、予測した段階ではなく警告の強さから段階を引き直す。
        spec = self._specs.get(name)
        if spec is None or not spec.inverted:
            return Dimension(name, score, level)
        alarm = alarm_of(spec, score)
        return Dimension(name, score, level_for(alarm, spec.levels), (), alarm, spec.alert_name)

    def _predict(self, model: Any, vector: Sequence[float]) -> tuple[Level, float]:
        level = _level_of(str(model.predict([vector])[0]))
        return level, self._severity(model, vector, level)

    def _severity(self, model: Any, vector: Sequence[float], level: Level) -> float:
        # 0..1 の重症度。確率が取れれば段階の期待値でならし、無ければ段階そのもの。
        proba = getattr(model, "predict_proba", None)
        classes = getattr(model, "classes_", None)
        if proba is None or classes is None:
            return int(level) / int(Level.HIGH)
        weights = proba([vector])[0]
        expected = sum(
            int(_level_of(str(c))) * float(w) for c, w in zip(classes, weights, strict=True)
        )
        return expected / int(Level.HIGH)


def load_bundle(path: str) -> dict:
    # ML 経路が選ばれたときだけ必要な重い依存なので、ここで遅延 import する。
    import joblib

    return joblib.load(path)
