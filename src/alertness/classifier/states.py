"""評価軸の仕様と、スコアからレベルへの変換。

Level / Dimension / Assessment / CueResult 自体は contracts に置いている。
ここは「何を判定するか」を表す DimensionSpec と、しきい値変換だけを持つ。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import Level


@dataclass(frozen=True)
class DimensionSpec:
    """1本の評価軸の設定。config の assessment.dimensions に対応する。"""

    name: str
    levels: dict[str, float]  # low / medium / high の境界
    cues: tuple[str, ...]  # この軸に効く cue 名


def level_for(score: float, levels: dict[str, float]) -> Level:
    # スコア(0..1)を段階に変換する。
    if score >= levels.get("high", 0.8):
        return Level.HIGH
    if score >= levels.get("medium", 0.6):
        return Level.MEDIUM
    if score >= levels.get("low", 0.3):
        return Level.LOW
    return Level.NONE
