"""うつむきの手がかり。中立姿勢より下を向き続ける＝居眠り・疲労の疑い。

pitch_rel はキャリブの中立姿勢からの差。符号は環境で反転しうるので、
もし上向きで反応してしまう場合は config の pitch_down_deg の符号運用を見直す。

目が開いたまま下を向いているのは、手元のスマホや資料を見ているのであって眠気ではない。
それは前方を見ていない状態として attention_buffer（注意散漫）が拾う。eyes_closed_ratio を
与えると、うつむいている間に目も閉じているときだけ眠気として立てる。
"""

from __future__ import annotations

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._support import eye_key, time_fraction, window_values


class HeadDownCue:
    name = "head_down"
    dimension = "drowsiness"

    def __init__(
        self,
        pitch_down_deg: float = 15.0,
        sustained_seconds: float = 1.5,
        eyes_closed_ratio: float = 0.0,
        eyes_closed_fraction: float = 0.5,
    ) -> None:
        self.pitch_down_deg = pitch_down_deg
        self.sustained_seconds = sustained_seconds
        self.eyes_closed_ratio = eyes_closed_ratio  # 0 なら目を見ない（頭の向きだけで判定）
        self.eyes_closed_fraction = eyes_closed_fraction  # 窓のうち目が閉じていた時間の下限

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし")

        times, pitches = window_values(obs, "pitch_rel", self.sustained_seconds, 0.0)
        if not pitches:
            return CueResult(self.name, self.dimension, 0.0, False, "")

        latest = pitches[-1]
        sustained = time_fraction(times, [p > self.pitch_down_deg for p in pitches])
        score = clamp(latest / self.pitch_down_deg) * sustained if self.pitch_down_deg > 0 else 0.0
        active = sustained >= 0.7 and latest > self.pitch_down_deg
        detail = f"下向き {latest:.0f}°"
        if score > 0 and self._eyes_open(obs, times, pitches, sustained):
            return CueResult(self.name, self.dimension, 0.0, False, f"{detail}（目は開いている）")
        return CueResult(self.name, self.dimension, score, active, detail)

    def _eyes_open(
        self, obs: Observation, times: list[float], pitches: list[float], down: float
    ) -> bool:
        """下を向いていた時間のうち、目も閉じていた割合が足りないか。

        窓全体の閉眼率で見ると、目を閉じてから目を開けて下を向いた場合も通ってしまう。
        同じフレームで両方が起きていた時間だけを数える（同じ履歴から読むので並びは揃う）。
        """
        if self.eyes_closed_ratio <= 0 or down <= 0:
            return False
        _, eyes = window_values(obs, eye_key(obs), self.sustained_seconds, 1.0)
        both = [
            p > self.pitch_down_deg and e < self.eyes_closed_ratio
            for p, e in zip(pitches, eyes, strict=True)
        ]
        return time_fraction(times, both) / down < self.eyes_closed_fraction
