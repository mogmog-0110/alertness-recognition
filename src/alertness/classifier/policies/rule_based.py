"""cue の結果を評価軸ごとに統合するルールベースの方針。

軸ごとに「重み付き平均」と「最も強い単独シグナル」の大きい方を取る。
弱い手がかりは足し合わせで効き、明確に強い手がかりは単独でも効く、という折衷。
EMA でならしてレベルのちらつきを抑える（ヒステリシス）。
"""

from __future__ import annotations

from collections.abc import Sequence

from ...contracts import Assessment, CueResult, Dimension, Observation
from ...geometry import clamp
from ..states import DimensionSpec, level_for


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
            score = self._smooth(spec.name, self._dimension_score(spec, results))
            contributing = tuple(r.name for r in results if r.active)
            level = level_for(score, spec.levels)
            dims[spec.name] = Dimension(spec.name, score, level, contributing)
        return Assessment(dimensions=dims, timestamp=obs.features.timestamp, cues=tuple(cues))

    def _dimension_score(self, spec: DimensionSpec, results: Sequence[CueResult]) -> float:
        if not results:
            return 0.0
        total_w = sum(self._weights.get(n, 1.0) for n in spec.cues) or 1.0
        weighted = sum(self._weights.get(r.name, 1.0) * r.score for r in results) / total_w
        strongest = max((r.score for r in results if r.active), default=0.0)
        return clamp(max(weighted, strongest))

    def _smooth(self, name: str, value: float) -> float:
        prev = self._ema.get(name, value)
        smoothed = self._alpha * value + (1.0 - self._alpha) * prev
        self._ema[name] = smoothed
        return smoothed
