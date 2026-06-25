"""テスト用のダミー生成ヘルパ。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from alertness.contracts import FaceLandmarks, Features, Frame, Observation


class FakeHistory:
    """History の代用。テストで履歴を差し込むために使う。"""

    def __init__(self, frames: Sequence[Features], fps: float = 30.0) -> None:
        self._frames = list(frames)
        self._fps = fps

    @property
    def fps(self) -> float:
        return self._fps

    def latest(self) -> Features | None:
        return self._frames[-1] if self._frames else None

    def recent(self, seconds: float) -> Sequence[Features]:
        if not self._frames:
            return []
        cutoff = self._frames[-1].timestamp - seconds
        return [f for f in self._frames if f.timestamp >= cutoff]


def make_observation(features: Features, history: FakeHistory | None = None) -> Observation:
    frame = Frame(image=np.zeros((2, 2, 3), dtype=np.uint8), index=0, timestamp=features.timestamp)
    landmarks = FaceLandmarks(points=np.zeros((0, 3)), image_size=(2, 2), detected=False)
    return Observation(
        frame=frame,
        landmarks=landmarks,
        features=features,
        history=history or FakeHistory([features]),
        profile=None,  # type: ignore[arg-type]  # policy/cue はここを参照しない
    )
