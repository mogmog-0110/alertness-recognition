"""閉眼の手がかり（PERCLOS）。眠気の主要シグナル。"""

from __future__ import annotations

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._eye_health import eye_signal_usable
from ._support import time_fraction, window_coverage, window_values


class EyeClosureCue:
    name = "eye_closure"
    dimension = "drowsiness"

    def __init__(
        self,
        window_seconds: float = 30.0,
        perclos_drowsy: float = 0.4,
        closed_ratio: float = 0.6,
        max_yaw: float = 25.0,
        min_coverage: float = 0.5,
        health_window: float = 60.0,
    ) -> None:
        self.window_seconds = window_seconds
        self.perclos_drowsy = perclos_drowsy  # この割合以上閉じていたら眠気とみなす
        self.closed_ratio = closed_ratio  # 開眼基準の何割未満で閉眼とするか
        self.max_yaw = max_yaw  # これ以上横を向くとEARが信用できないので判定しない
        self.min_coverage = min_coverage  # 窓のうち顔が見えていた時間がこれ未満なら判定しない
        self.health_window = health_window  # この長さに瞬きが1回も無ければ目の信号を信じない

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし", None, False)
        if abs(obs.features.get("yaw_rel", 0.0)) > self.max_yaw:
            # 横顔ではEARが壊れて誤検出するので、眠気判定から除外する。
            return CueResult(self.name, self.dimension, 0.0, False, "横向き", None, False)

        coverage = window_coverage(obs, self.window_seconds)
        if coverage < self.min_coverage:
            # 窓の大半で顔を見失っている。残った少数のフレームで出した PERCLOS は、
            # 窓全体の閉眼割合を表していない。
            detail = f"計測不足 {coverage:.0%}"
            return CueResult(self.name, self.dimension, 0.0, False, detail, None, False)

        usable, reason = eye_signal_usable(obs, self.health_window, closed_ratio=self.closed_ratio)
        if not usable:
            # サングラス・暗所では EAR が低いまま張り付き、閉じていないのに PERCLOS が
            # 上がる。眠気を誤って警告する向きの誤りなので、黙って頭部の cue に譲る。
            return CueResult(self.name, self.dimension, 0.0, False, reason, None, False)

        times, ears = window_values(obs, "ear_norm", self.window_seconds, 1.0)
        flags = [e < self.closed_ratio for e in ears]
        perclos = time_fraction(times, flags)
        score = clamp(perclos / self.perclos_drowsy) if self.perclos_drowsy > 0 else 0.0
        active = perclos >= self.perclos_drowsy
        return CueResult(self.name, self.dimension, score, active, f"PERCLOS {perclos:.2f}")
