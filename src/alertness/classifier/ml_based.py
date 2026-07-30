"""学習済みモデル(model.pkl)を読んで判定する Classifier 実装。

ルールベースの CueClassifier の隣に置く、差し替え可能なもう一つの判定器。
cue は使わず、features を「学習時に保存した列順」でベクトル化し、軸ごとの
モデルで段階を予測する。学習(alertness-colab)が書き出した bundle
{models, features, classes} をそのまま受け取る。モデルはフレーム単位で判定する
scikit-learn 由来（SVM 等）。学習と推論で同じ列順・同じ軸名になるのが唯一の取り決め。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import Assessment, Dimension, Features, Level, Observation
from .states import DimensionSpec, alarm_of, level_for

# 学習のターゲット列 "label_<軸>" と、本体の評価軸名 "<軸>" をつなぐ接頭辞。
_AXIS_PREFIX = "label_"

# 「その特徴量に値があったか」を表す列の接尾辞。学習側(alertness-colab)と同じ規約。
# rPPG のように欠けるのが普通の特徴は、欠損を 0 で埋めるだけだと心拍0bpm のような
# 実在しない値になるので、値の有無そのものを別の特徴として渡している。
_PRESENT_SUFFIX = "_present"

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
        # 学習が安静基準で中心化されているか。されているなら推論も同じ中心化を通す。
        # 学習側(rest_basis)と推論側(profile.baselines)で対称に「安静からの差」にする。
        self._rest_centered = bool(bundle.get("rest_centered"))
        # 軸の向き（高いほど良い軸か）は config 側の取り決めなので、rule と同じ spec を使う。
        self._specs = {s.name: s for s in (dimensions or ())}

    def assess(self, obs: Observation) -> Assessment:
        baselines = obs.profile.baselines if self._rest_centered else {}
        sample = self._vector(obs.features, baselines)
        dims: dict[str, Dimension] = {}
        for target, model in self._models.items():
            name = _dimension_name(target)
            level, score = self._predict(model, sample)
            dims[name] = self._as_dimension(name, score, level)
        return Assessment(dimensions=dims, timestamp=obs.features.timestamp)

    def _vector(self, features: Features, baselines: Mapping[str, float]) -> list[float]:
        # 欠損は 0.0（学習側の fillna(0.0) と揃える）。列順は bundle に従う。
        return [self._value(features, name, baselines) for name in self._features]

    def _value(self, features: Features, name: str, baselines: Mapping[str, float]) -> float:
        if name.endswith(_PRESENT_SUFFIX):
            base = name[: -len(_PRESENT_SUFFIX)]
            return 0.0 if math.isnan(features.get(base, float("nan"))) else 1.0
        value = features.get(name, 0.0)
        if math.isnan(value):
            return 0.0
        # 学習が安静中心化されていれば、その人の安静基準を引いて同じ空間にそろえる。
        return value - baselines.get(name, 0.0)

    def _as_dimension(self, name: str, score: float, level: Level) -> Dimension:
        # 反転する軸（集中など）は、予測した段階ではなく警告の強さから段階を引き直す。
        spec = self._specs.get(name)
        if spec is None or not spec.inverted:
            return Dimension(name, score, level)
        alarm = alarm_of(spec, score)
        return Dimension(name, score, level_for(alarm, spec.levels), (), alarm, spec.alert_name)

    def _predict(self, model: Any, sample: Sequence) -> tuple[Level, float]:
        level = _level_of(str(model.predict([sample])[0]))
        return level, self._severity(model, sample, level)

    def _severity(self, model: Any, sample: Sequence, level: Level) -> float:
        # 0..1 の重症度。確率が取れれば段階の期待値でならし、無ければ段階そのもの。
        proba = getattr(model, "predict_proba", None)
        classes = getattr(model, "classes_", None)
        if proba is None or classes is None:
            return int(level) / int(Level.HIGH)
        weights = proba([sample])[0]
        expected = sum(
            int(_level_of(str(c))) * float(w) for c, w in zip(classes, weights, strict=True)
        )
        return expected / int(Level.HIGH)


def load_bundle(path: str) -> dict:
    # ML 経路が選ばれたときだけ必要な重い依存なので、ここで遅延 import する。
    import joblib

    return joblib.load(path)
