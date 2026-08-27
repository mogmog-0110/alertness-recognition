"""評価軸の仕様と、スコアからレベルへの変換。

Level / Dimension / Assessment / CueResult 自体は contracts に置いている。
ここは「何を判定するか」を表す DimensionSpec と、しきい値変換だけを持つ。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import Level

ALERT_ON = ("high", "low")
COMBINE = ("max", "weighted")

# 段を上げるしきい値と下げるしきい値の差（警告の強さの単位）。
DEFAULT_RELEASE_MARGIN = 0.08


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
    # combine=weighted のとき、警告を満額にするのに要る「兆候あり」の cue 本数。
    # これを下回る本数しか同意していなければ、その割合まで警告を割り引く。
    min_agree: int = 2
    # 段を下げるのに要る、いまの段の入口からの余裕。0 なら従来どおり境界1本で上下する。
    release_margin: float = DEFAULT_RELEASE_MARGIN

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


def entry_threshold(level: Level, levels: dict[str, float]) -> float:
    """その段に入るのに要る警告の強さ。NONE は下限が無いので 0。"""
    if level == Level.HIGH:
        return levels.get("high", 0.8)
    if level == Level.MEDIUM:
        return levels.get("medium", 0.6)
    if level == Level.LOW:
        return levels.get("low", 0.3)
    return 0.0


class LevelLatch:
    """段の上げ下げに別々のしきい値を使い、境界付近のばたつきを止める。

    上げは levels の境界そのまま（危険側は待たせない）、下げは入った境界を
    release_margin ぶん下回るまで待つ。値が境界の上下を往復するあいだ、
    段は上のまま留まるので、警告音が細かく鳴り止み鳴り直すのを防ぐ。

    EMA だけでは代わりにならない。EMA は値の凹凸をならすが、ならした値が境界を
    またぐこと自体は止められず、境界に張り付いた入力では往復がそのまま残る。
    """

    def __init__(self, levels: dict[str, float], release_margin: float) -> None:
        self._levels = dict(levels)
        self._margin = max(0.0, release_margin)
        self._level = Level.NONE

    @property
    def level(self) -> Level:
        return self._level

    def reset(self) -> None:
        self._level = Level.NONE

    def update(self, alarm: float) -> Level:
        raised = level_for(alarm, self._levels)
        if raised > self._level:
            self._level = raised
            return self._level
        # 下げるのは、いまの段の入口より margin ぶん下回ってから。一気に複数段
        # 落ちることもあるので、条件を満たさなくなるまで繰り返す。
        while self._level > Level.NONE:
            release = entry_threshold(self._level, self._levels) - self._margin
            if alarm >= release:
                break
            self._level = Level(int(self._level) - 1)
        return self._level
