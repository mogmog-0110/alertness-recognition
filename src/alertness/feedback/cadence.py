"""警告音を「いま鳴らすか」だけを決める。鳴らし方（PC のスピーカーか端末か）は持たない。

PC の窓と端末のブラウザで同じ間合いにするため、判断をここに一本化する。
MEDIUM は一定間隔、HIGH は半分の間隔から始めて鳴らすたびに詰める。無視され続けて
いるのに間隔が変わらないと、そのまま気づかれずに終わる。詰めっぱなしはうるさくて
装置ごと切られるので下限を置く。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..contracts import Level


@dataclass(frozen=True)
class _Episode:
    """1回の警告のまとまり。段が NONE/LOW まで下がるまで続く。"""

    level: Level
    plays: int  # この警告で何回鳴らしたか。間隔を詰める根拠
    last_at: float


class AlertCadence:
    """軸ごとに独立して、鳴らす時点を決める。"""

    def __init__(
        self,
        cooldown_seconds: float = 5.0,
        min_interval_seconds: float = 1.5,
        escalate_factor: float = 0.7,
    ) -> None:
        self._cooldown = cooldown_seconds  # 注意喚起(MEDIUM)の間隔
        self._min_interval = min_interval_seconds  # どれだけ詰めてもこれより短くしない
        self._escalate = escalate_factor  # 警告(HIGH)が続く間、間隔にかける係数
        self._episodes: dict[str, _Episode] = {}

    @property
    def min_interval(self) -> float:
        return self._min_interval

    def due(self, name: str, level: Level, now: float) -> bool:
        """その軸の現在の段を伝え、鳴らし時なら鳴らしたことにして True を返す。

        段に関わらず毎フレーム呼ぶこと。収まったことも伝わらないと、次に立ったときに
        「続きの警告」と誤解して詰めた間隔から鳴り始める。
        """
        if not self.pending(name, level, now):
            return False
        self.mark(name, level, now)
        return True

    def pending(self, name: str, level: Level, now: float) -> bool:
        """いま鳴らし時か。鳴らしたことにはしない（複数の軸から 1 つだけ選ぶとき用）。"""
        if level < Level.MEDIUM:
            self._episodes.pop(name, None)  # 収まった。次は最初の1回から数え直す
            return False
        episode = self._episodes.get(name)
        if episode is None or level > episode.level:
            return True  # 立った直後と、段が上がった直後は待たせない
        if now - episode.last_at < self._interval(level, episode.plays):
            self._episodes[name] = replace(episode, level=level)
            return False
        return True

    def mark(self, name: str, level: Level, now: float) -> None:
        """その軸を鳴らした。pending が True を返した直後に呼ぶ。"""
        episode = self._episodes.get(name)
        if episode is None or level > episode.level:
            self._episodes[name] = _Episode(level, 1, now)
        else:
            self._episodes[name] = _Episode(level, episode.plays + 1, now)

    def reset(self) -> None:
        self._episodes.clear()

    def _interval(self, level: Level, plays: int) -> float:
        if level < Level.HIGH:
            return self._cooldown  # 注意喚起は一定間隔で。急かす段ではない
        base = self._cooldown / 2.0
        return max(self._min_interval, base * self._escalate ** max(0, plays - 1))
