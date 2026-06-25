"""うつむきの手がかり。中立姿勢より下を向き続ける＝居眠り・疲労の疑い。

pitch_rel はキャリブの中立姿勢からの差。符号は環境で反転しうるので、
もし上向きで反応してしまう場合は config の pitch_down_deg の符号運用を見直す。
"""

from __future__ import annotations

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._support import true_fraction, window_values


class HeadDownCue:
    name = "head_down"
    dimension = "drowsiness"

    def __init__(self, pitch_down_deg: float = 15.0, sustained_seconds: float = 1.5) -> None:
        self.pitch_down_deg = pitch_down_deg
        self.sustained_seconds = sustained_seconds

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし")

        _, pitches = window_values(obs, "pitch_rel", self.sustained_seconds, 0.0)
        if not pitches:
            return CueResult(self.name, self.dimension, 0.0, False, "")

        latest = pitches[-1]
        sustained = true_fraction([p > self.pitch_down_deg for p in pitches])
        score = clamp(latest / self.pitch_down_deg) * sustained if self.pitch_down_deg > 0 else 0.0
        active = sustained >= 0.7 and latest > self.pitch_down_deg
        return CueResult(self.name, self.dimension, score, active, f"下向き {latest:.0f}°")
