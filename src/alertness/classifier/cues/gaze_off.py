"""画面外注視の手がかり。視線が基準位置から外れ続ける＝注意逸脱の疑い。"""

from __future__ import annotations

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._support import trailing_true_seconds, window_values


class GazeOffCue:
    name = "gaze_off"
    dimension = "distraction"

    def __init__(self, off_threshold: float = 0.2, off_screen_seconds: float = 2.0) -> None:
        self.off_threshold = off_threshold  # 基準位置からこれ以上ズレたら画面外
        self.off_screen_seconds = off_screen_seconds

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし")

        window = max(2.0, self.off_screen_seconds * 2)
        times, offs = window_values(obs, "gaze_off", window, 0.0)
        flags = [o > self.off_threshold for o in offs]
        duration = trailing_true_seconds(times, flags)
        score = clamp(duration / self.off_screen_seconds) if self.off_screen_seconds > 0 else 0.0
        active = duration >= self.off_screen_seconds
        return CueResult(self.name, self.dimension, score, active, f"視線外 {duration:.1f}s")
