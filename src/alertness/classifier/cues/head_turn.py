"""よそ見の手がかり。中立より横を向き続ける＝注意逸脱の疑い。"""

from __future__ import annotations

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._support import true_fraction, window_values


class HeadTurnCue:
    name = "head_turn"
    dimension = "distraction"

    def __init__(self, yaw_side_deg: float = 25.0, sustained_seconds: float = 1.5) -> None:
        self.yaw_side_deg = yaw_side_deg
        self.sustained_seconds = sustained_seconds

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし")

        _, yaws = window_values(obs, "yaw_rel", self.sustained_seconds, 0.0)
        if not yaws:
            return CueResult(self.name, self.dimension, 0.0, False, "")

        latest = abs(yaws[-1])
        sustained = true_fraction([abs(y) > self.yaw_side_deg for y in yaws])
        score = clamp(latest / self.yaw_side_deg) * sustained if self.yaw_side_deg > 0 else 0.0
        active = sustained >= 0.7 and latest > self.yaw_side_deg
        return CueResult(self.name, self.dimension, score, active, f"横向き {yaws[-1]:.0f}°")
