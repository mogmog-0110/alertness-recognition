"""よそ見の手がかり。中立より横を向き続ける＝注意逸脱の疑い。"""

from __future__ import annotations

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._departure import TURN, departure
from ._support import time_fraction, window_values


class HeadTurnCue:
    name = "head_turn"
    dimension = "distraction"

    def __init__(
        self,
        yaw_side_deg: float = 25.0,
        sustained_seconds: float = 1.5,
        lost_hold_seconds: float = 0.0,
    ) -> None:
        self.yaw_side_deg = yaw_side_deg
        self.sustained_seconds = sustained_seconds
        # 横を向きすぎて顔を見失っても、この秒数までは横を向いたままとみなす。0 なら見ない。
        # 顔が外れると向きが測れず、脇見が注意散漫や見失いとして鳴ってしまう。
        self.lost_hold_seconds = lost_hold_seconds

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return self._while_lost(obs)

        times, yaws = window_values(obs, "yaw_rel", self.sustained_seconds, 0.0)
        if not yaws:
            return CueResult(self.name, self.dimension, 0.0, False, "")

        latest = abs(yaws[-1])
        sustained = time_fraction(times, [abs(y) > self.yaw_side_deg for y in yaws])
        score = clamp(latest / self.yaw_side_deg) * sustained if self.yaw_side_deg > 0 else 0.0
        active = sustained >= 0.7 and latest > self.yaw_side_deg
        return CueResult(self.name, self.dimension, score, active, f"横向き {yaws[-1]:.0f}°")

    def _while_lost(self, obs: Observation) -> CueResult:
        if self.lost_hold_seconds <= 0:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし")
        gone = departure(obs, self.yaw_side_deg, 0.0)
        if gone is None or gone.kind != TURN or gone.absent_seconds > self.lost_hold_seconds:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし")
        turned = gone.moved_seconds + gone.absent_seconds
        score = clamp(turned / self.sustained_seconds) if self.sustained_seconds > 0 else 1.0
        active = turned >= self.sustained_seconds
        return CueResult(self.name, self.dimension, score, active, "横を向いて顔が外れた")
