"""集中の手がかり。視線が画面に載り、頭部も安定した状態が続く＝集中の疑い。

注意逸脱(gaze_off/head_turn)の裏返しに近いが、「載り続けている」ことを正の側で測る独立の
手がかりにする。短い注視ではなく、持続を重く見る（末尾から連続で載っている時間で採点）。
生体信号もモデルも要らないので、rule 経路のまま手元のカメラで挙動を確かめられる。
"""

from __future__ import annotations

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._support import trailing_true_seconds, window_values


class AttentionHoldCue:
    name = "attention_hold"
    dimension = "concentration"

    def __init__(
        self,
        gaze_on_threshold: float = 0.035,
        steady_yaw_deg: float = 12.0,
        sustained_seconds: float = 3.0,
    ) -> None:
        self.gaze_on_threshold = gaze_on_threshold  # これ以下なら視線が画面に載っている
        self.steady_yaw_deg = steady_yaw_deg  # これ以下なら頭部が安定
        self.sustained_seconds = sustained_seconds  # これだけ続けば満点

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし")

        window = max(2.0, self.sustained_seconds * 2)
        # 特徴が無いフレームは「載っていない」側へ倒す（gaze大・yaw大を既定に）。
        times, gaze = window_values(obs, "gaze_off", window, 1.0)
        _, yaws = window_values(obs, "yaw_rel", window, 90.0)
        if not times:
            return CueResult(self.name, self.dimension, 0.0, False, "")

        on = [
            g <= self.gaze_on_threshold and abs(y) <= self.steady_yaw_deg
            for g, y in zip(gaze, yaws, strict=False)
        ]
        held = trailing_true_seconds(times, on)
        # 起動直後は履歴が sustained_seconds に満たない。そこを満点扱いにも 0 扱いにもせず、
        # 「見えている範囲での保持率」で採点する（集中は低いほど警告する軸なので、履歴不足を
        # 0＝集中していない と読むと起動直後に必ず誤警告になる）。
        span = times[-1] - times[0]
        required = min(self.sustained_seconds, span) if span > 0 else 0.0
        score = clamp(held / required) if required > 0 else 1.0
        active = held >= self.sustained_seconds
        return CueResult(self.name, self.dimension, score, active, f"注視保持 {held:.1f}s")
