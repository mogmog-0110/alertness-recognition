"""あくびの手がかり。口の開きが一定時間続いたら疲労・眠気のサイン。

blendshape の jawOpen があれば優先して使い、無ければ MAR で代用する。
"""

from __future__ import annotations

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._support import trailing_true_seconds, window_values


class YawnCue:
    name = "yawn"
    dimension = "drowsiness"

    def __init__(self, open_threshold: float = 0.5, min_seconds: float = 1.5) -> None:
        self.open_threshold = open_threshold  # 口が開いているとみなす値
        self.min_seconds = min_seconds  # この時間続いたらあくびとみなす

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし")

        key = "jawOpen" if "jawOpen" in obs.features.values else "mar"
        window = max(2.0, self.min_seconds * 2)
        times, values = window_values(obs, key, window, 0.0)
        flags = [v >= self.open_threshold for v in values]
        duration = trailing_true_seconds(times, flags)
        score = clamp(duration / self.min_seconds) if self.min_seconds > 0 else 0.0
        active = duration >= self.min_seconds
        return CueResult(self.name, self.dimension, score, active, f"あくび {duration:.1f}s")
