"""cue の結果を評価軸ごとに統合するルールベースの方針。

軸ごとに「重み付き平均」と「最も強い単独シグナル」の大きい方を取る。
弱い手がかりは足し合わせで効き、明確に強い手がかりは単独でも効く、という折衷。
EMA でならしてレベルのちらつきを抑える（ヒステリシス）。

統合はすべて「警告の強さ」の空間で行う。集中のように高いほど良い軸は cue のスコアを
先に反転してから足す。スコアのまま max を取ると、良い方の cue が勝って警告が鈍るため。
"""

from __future__ import annotations

from collections.abc import Sequence

from ...contracts import Assessment, CueResult, Dimension, Observation
from ...geometry import clamp
from ..states import DimensionSpec, alarm_of, level_for


class RuleBasedPolicy:
    def __init__(
        self,
        dimensions: Sequence[DimensionSpec],
        weights: dict[str, float],
        hysteresis_frames: int = 8,
    ) -> None:
        self._dims = tuple(dimensions)
        self._weights = dict(weights)
        self._alpha = 2.0 / (max(1, hysteresis_frames) + 1)  # EMA係数
        self._ema: dict[str, float] = {}

    def decide(self, obs: Observation, cues: Sequence[CueResult]) -> Assessment:
        by_dim: dict[str, list[CueResult]] = {}
        for r in cues:
            by_dim.setdefault(r.dimension, []).append(r)

        dims: dict[str, Dimension] = {}
        for spec in self._dims:
            results = by_dim.get(spec.name, [])
            alarm = self._smooth(spec.name, self._dimension_alarm(spec, results))
            score = alarm_of(spec, alarm)  # 反転は自己逆写像なので同じ関数で軸の値に戻る
            contributing = tuple(r.name for r in results if r.active)
            dims[spec.name] = Dimension(
                spec.name,
                score,
                level_for(alarm, spec.levels),
                contributing,
                alarm if spec.inverted else None,
                spec.alert_name,
            )
        return Assessment(dimensions=dims, timestamp=obs.features.timestamp, cues=tuple(cues))

    def _dimension_alarm(self, spec: DimensionSpec, results: Sequence[CueResult]) -> float:
        # 手がかりが1つも無い＝警告する理由が無い。反転軸でもここは 0（無言）にする。
        if not results:
            return 0.0
        total_w = sum(self._weights.get(n, 1.0) for n in spec.cues) or 1.0
        alarms = [(r, alarm_of(spec, r.score)) for r in results]
        weighted = sum(self._weights.get(r.name, 1.0) * a for r, a in alarms) / total_w
        if spec.combine == "weighted":
            # 手がかりの一致を要求する軸。単独で強く出ても、他が黙っていれば伸びない。
            return clamp(weighted)
        strongest = max((a for r, a in alarms if r.active), default=0.0)
        return clamp(max(weighted, strongest))

    def _smooth(self, name: str, value: float) -> float:
        prev = self._ema.get(name, value)
        smoothed = self._alpha * value + (1.0 - self._alpha) * prev
        self._ema[name] = smoothed
        return smoothed
