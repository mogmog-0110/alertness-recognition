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

    def recent(self, seconds: float) -> Sequence[Features]:
        # 末尾から seconds 秒ぶんを返す。
        if not self._buf:
            return []
        cutoff = self._buf[-1].timestamp - seconds
        return [f for f in self._buf if f.timestamp >= cutoff]
