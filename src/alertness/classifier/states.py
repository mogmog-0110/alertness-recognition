"""評価軸の仕様と、スコアからレベルへの変換。

Level / Dimension / Assessment / CueResult 自体は contracts に置いている。
ここは「何を判定するか」を表す DimensionSpec と、しきい値変換だけを持つ。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import Level

ALERT_ON = ("high", "low")
COMBINE = ("max", "weighted")


@dataclass(frozen=True)
class DimensionSpec:
    """1本の評価軸の設定。config の assessment.dimensions に対応する。"""

    name: str
    levels: dict[str, float]  # low / medium / high の境界
    cues: tuple[str, ...]  # この軸に効く cue 名
    alert_on: str = "high"  # high=スコアが高いほど警告 / low=低いほど警告（集中など）
    alert_name: str = ""  # 警告としての表示名。空なら name
    # max = 強い手がかりが1つあれば立てる（眠気のように、単独で決定的な兆候がある軸）
    # weighted = 重み付き平均だけを見る（複数の手がかりの一致を要求する軸）
    combine: str = "max"

    @property
    def inverted(self) -> bool:
        return self.alert_on == "low"


def alarm_of(spec: DimensionSpec, score: float) -> float:
    """軸のスコアを「警告の強さ」に変換する。高いほど良い軸（集中など）は反転する。"""
    return 1.0 - score if spec.inverted else score


def level_for(score: float, levels: dict[str, float]) -> Level:
    # スコア(0..1)を段階に変換する。
    if score >= levels.get("high", 0.8):
        return Level.HIGH
    if score >= levels.get("medium", 0.6):
        return Level.MEDIUM
    if score >= levels.get("low", 0.3):
        return Level.LOW
    return Level.NONE
