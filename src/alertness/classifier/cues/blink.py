"""長い閉眼の手がかり。連続して閉じ続ける＝マイクロスリープの疑い。"""

from __future__ import annotations

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._support import trailing_true_seconds, window_values


class BlinkCue:
    name = "blink"
    dimension = "drowsiness"

    def __init__(
        self, closed_ratio: float = 0.6, long_blink_seconds: float = 1.0, max_yaw: float = 25.0
    ) -> None:
        self.closed_ratio = closed_ratio
        self.long_blink_seconds = long_blink_seconds  # これ以上閉じ続けたら危険
        self.max_yaw = max_yaw  # 横向きはEARが信用できないので判定しない

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし")
        if abs(obs.features.get("yaw_rel", 0.0)) > self.max_yaw:
            return CueResult(self.name, self.dimension, 0.0, False, "横向き")

        window = max(2.0, self.long_blink_seconds * 3)
        times, ears = window_values(obs, "ear_norm", window, 1.0)
        flags = [e < self.closed_ratio for e in ears]
        duration = trailing_true_seconds(times, flags)
        score = clamp(duration / self.long_blink_seconds) if self.long_blink_seconds > 0 else 0.0
        active = duration >= self.long_blink_seconds
        return CueResult(self.name, self.dimension, score, active, f"閉眼 {duration:.1f}s")
