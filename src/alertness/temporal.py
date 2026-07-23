"""特徴量の時系列バッファ。

History を満たし、cue（PERCLOS・瞬きなど）や将来の時系列モデルが過去フレームを
読む口になる。リングバッファなので append で古いものから捨てる。
判定結果は不変だが、この履歴自体は性質上ためていく状態を持つ。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from .contracts import Features


class TemporalContext:
    def __init__(self, max_seconds: float = 60.0, fps: float = 30.0) -> None:
        self._fps = fps
        self._buf: deque[Features] = deque(maxlen=max(1, int(max_seconds * fps)))

    @property
    def fps(self) -> float:
        return self._fps

    def append(self, features: Features) -> None:
        self._buf.append(features)

    def latest(self) -> Features | None:
        return self._buf[-1] if self._buf else None

    @property
    def measured_fps(self) -> float:
        """実際に流れているフレームレート。要求値ではなく達成値を見るために使う。"""
        if len(self._buf) < 2:
            return self._fps
        span = self._buf[-1].timestamp - self._buf[0].timestamp
        return (len(self._buf) - 1) / span if span > 0 else self._fps

    def recent(self, seconds: float) -> Sequence[Features]:
        # 末尾から seconds 秒ぶんを返す。時刻は単調増加なので、古い側は見ずに末尾から
        # 遡って打ち切る。高フレームレートだとバッファが大きくなり、毎フレーム全走査すると
        # cue の呼び出し回数ぶん効いてくるため。
        if not self._buf:
            return []
        cutoff = self._buf[-1].timestamp - seconds
        out: list[Features] = []
        for features in reversed(self._buf):
            if features.timestamp < cutoff:
                break
            out.append(features)
        out.reverse()
        return out
