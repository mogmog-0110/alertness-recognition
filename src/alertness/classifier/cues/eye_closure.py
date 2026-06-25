"""閉眼の手がかり（PERCLOS）。眠気の主要シグナル。"""

from __future__ import annotations

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._support import true_fraction, window_values


class EyeClosureCue:
    name = "eye_closure"
    dimension = "drowsiness"

    def __init__(
        self,
        window_seconds: float = 30.0,
        perclos_drowsy: float = 0.4,
        closed_ratio: float = 0.6,
        max_yaw: float = 25.0,
    ) -> None:
        self.window_seconds = window_seconds
        self.perclos_drowsy = perclos_drowsy  # この割合以上閉じていたら眠気とみなす
        self.closed_ratio = closed_ratio  # 開眼基準の何割未満で閉眼とするか
        self.max_yaw = max_yaw  # これ以上横を向くとEARが信用できないので判定しない

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし")
        if abs(obs.features.get("yaw_rel", 0.0)) > self.max_yaw:
            # 横顔ではEARが壊れて誤検出するので、眠気判定から除外する。
            return CueResult(self.name, self.dimension, 0.0, False, "横向き")

        _, ears = window_values(obs, "ear_norm", self.window_seconds, 1.0)
        flags = [e < self.closed_ratio for e in ears]
        perclos = true_fraction(flags)
        score = clamp(perclos / self.perclos_drowsy) if self.perclos_drowsy > 0 else 0.0
        active = perclos >= self.perclos_drowsy
        return CueResult(self.name, self.dimension, score, active, f"PERCLOS {perclos:.2f}")
