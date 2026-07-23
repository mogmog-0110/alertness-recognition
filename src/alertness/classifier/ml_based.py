"""学習済みモデル(model.pkl)を読んで判定する Classifier 実装。

ルールベースの CueClassifier の隣に置く、差し替え可能なもう一つの判定器。
cue は使わず、features を「学習時に保存した列順」でベクトル化し、軸ごとの
モデルで段階を予測する。学習(alertness-colab)が書き出した bundle
{models, features, classes, window} をそのまま受け取る。学習と推論で同じ列順・同じ
軸名になるのが唯一の取り決め。

## 時系列モデル（LSTM 等）への対応

bundle の window が入力の形を決める。SVM や Random Forest のようにフレーム単位で
判定するモデルは window=1（省略時の既定）で、1フレーム分の特徴量ベクトルを渡す。
LSTM のように時系列窓を要るモデルは window>1 で、直近 window フレームを古い順に
並べた行列を渡す。過去フレームは Observation.history から取る（contracts.History が
「将来 LSTM などが時系列窓を読む口にもなる」として用意されている口）。

窓の長さを推論側で決め打ちせず学習成果物に持たせるのは、列順(features)と同じ理由。
学習と推論で食い違うと静かに壊れるので、取り決めは model.pkl 側に一本化する。

時系列モデルは pickle されたクラスではなく素のデータとして入っている（学習側のコードを
import せずに読めるようにするため）。復元は sequence_model が担う。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import Assessment, Dimension, Features, Level, Observation
from .sequence_model import load_models, window_of
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

# 履歴を要求する時間に掛ける余裕。実フレームレートが公称より低いと window 枚に足りない
# ので、多めに取り出してから末尾を切る。取りすぎても末尾を切るだけで害はない。
_HISTORY_MARGIN = 2.0


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
        # 時系列モデルはクラスではなく素のデータとして入っているので、ここで復元する
        # （学習側のコードを import せずに読めるようにするため。sequence_model 参照）。
        self._models = load_models(models)
        self._features = list(features)
        # 入力の形。1 ならフレーム単位、2以上なら直近 window フレームの行列を渡す。
        self._window = max(1, int(bundle.get("window") or window_of(self._models)))
        # 学習が安静基準で中心化されているか。されているなら推論も同じ中心化を通す。
        # 学習側(rest_basis)と推論側(profile.baselines)で対称に「安静からの差」にする。
        self._rest_centered = bool(bundle.get("rest_centered"))
        # 軸の向き（高いほど良い軸か）は config 側の取り決めなので、rule と同じ spec を使う。
        self._specs = {s.name: s for s in (dimensions or ())}

    @property
    def window(self) -> int:
        return self._window

    def assess(self, obs: Observation) -> Assessment:
        sample = self._sample(obs)
        dims: dict[str, Dimension] = {}
        for target, model in self._models.items():
            name = _dimension_name(target)
            level, score = self._predict(model, sample)
            dims[name] = self._as_dimension(name, score, level)
        return Assessment(dimensions=dims, timestamp=obs.features.timestamp)

    def _sample(self, obs: Observation) -> list:
        """モデルへ渡す1件分の入力。window=1 なら特徴量ベクトル、2以上ならその行列。"""
        baselines = obs.profile.baselines if self._rest_centered else {}
        if self._window == 1:
            return self._vector(obs.features, baselines)
        return self._sequence(obs, baselines)

    def _sequence(self, obs: Observation, baselines: Mapping[str, float]) -> list[list[float]]:
        """直近 window フレームの特徴量を古い順に並べる。

        履歴が足りない起動直後は、最も古い行を複製して前に詰める。判定を止めてしまうと、
        窓が満ちるまでの数秒間だけ挙動が変わり、ルール経路との比較がしづらくなるため。
        """
        fps = obs.history.fps if obs.history.fps > 0 else 30.0
        span = self._window / fps * _HISTORY_MARGIN
        rows = [self._vector(f, baselines) for f in obs.history.recent(span)][-self._window :]
        if not rows:
            rows = [self._vector(obs.features, baselines)]
        return [rows[0]] * (self._window - len(rows)) + rows

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
