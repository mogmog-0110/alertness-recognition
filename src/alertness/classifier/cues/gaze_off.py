"""画面外注視の手がかり。視線が基準位置から外れ続ける＝注意逸脱の疑い。

虹彩位置から視線を測るので、目を細めたり閉じたりすると虹彩点の追跡が乱れ、
実際は前を向いたままでも視線が外れたように見える（あくびで目を閉じたときに
脇見と誤検出していた）。eyes_closed_ratio を与えると、その間は「外れている」と
数えない。目を見ないと下げるのではなく、そのフレームだけ判定から外す。
"""

from __future__ import annotations

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._support import eye_key, trailing_true_seconds, window_values


class GazeOffCue:
    name = "gaze_off"
    dimension = "distraction"

    def __init__(
        self,
        off_threshold: float = 0.2,
        off_screen_seconds: float = 2.0,
        eyes_closed_ratio: float = 0.0,
    ) -> None:
        self.off_threshold = off_threshold  # 基準位置からこれ以上ズレたら画面外
        self.off_screen_seconds = off_screen_seconds
        self.eyes_closed_ratio = eyes_closed_ratio  # 0 なら目を見ない（虹彩の値をそのまま使う）

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし")

        window = max(2.0, self.off_screen_seconds * 2)
        times, offs = window_values(obs, "gaze_off", window, 0.0)
        flags = [o > self.off_threshold for o in offs]
        if self.eyes_closed_ratio > 0:
            _, eyes = window_values(obs, eye_key(obs), window, 1.0)
            flags = [
                flag and eye >= self.eyes_closed_ratio
                for flag, eye in zip(flags, eyes, strict=True)
            ]
        duration = trailing_true_seconds(times, flags)
        score = clamp(duration / self.off_screen_seconds) if self.off_screen_seconds > 0 else 0.0
        active = duration >= self.off_screen_seconds
        return CueResult(self.name, self.dimension, score, active, f"視線外 {duration:.1f}s")
