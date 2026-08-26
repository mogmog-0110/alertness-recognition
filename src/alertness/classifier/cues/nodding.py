"""うなずきの手がかり。首の力が抜けて頭が落ち、はっと戻す動き。

head_down が見ているのは「下を向いたまま続いている」状態で、持続を条件にしているため、
落ちてすぐ戻す動きは窓の中の一瞬にしかならず消える。しかし居眠りの入り口で最初に出るのは
その往復の方で、姿勢が下を向いたまま固まるのはもっと後になる。両方が要る。

数えるのは「速く落ちて、短い時間で戻った」回数。戻らずに下を向いたままなら、それは
うなずきではなく居眠り姿勢なので head_down に任せてここでは数えない。
"""

from __future__ import annotations

from statistics import median

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._support import window_coverage, window_values


class NoddingCue:
    name = "nodding"
    dimension = "drowsiness"

    def __init__(
        self,
        window_seconds: float = 60.0,
        amplitude_deg: float = 8.0,
        max_seconds: float = 2.5,
        nods_drowsy: int = 3,
        min_samples: int = 10,
        min_coverage: float = 0.5,
    ) -> None:
        self.window_seconds = window_seconds  # うなずきを数える窓
        self.amplitude_deg = amplitude_deg  # この角度ぶん落ちたら1回とみなす
        self.max_seconds = max_seconds  # これより長く戻らなければ「居眠り姿勢」で数えない
        self.nods_drowsy = nods_drowsy  # 窓内でこの回数に達したら満点
        self.min_samples = min_samples  # 基準を取るのに要る標本数
        self.min_coverage = min_coverage  # 窓のうち顔が見えていた時間の下限

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし", None, False)

        coverage = window_coverage(obs, self.window_seconds)
        if coverage < self.min_coverage:
            detail = f"計測不足 {coverage:.0%}"
            return CueResult(self.name, self.dimension, 0.0, False, detail, None, False)

        times, pitches = window_values(obs, "pitch_rel", self.window_seconds, 0.0)
        if len(pitches) < self.min_samples:
            return CueResult(self.name, self.dimension, 0.0, False, "姿勢を観察中", None, False)

        nods = self._count(times, pitches)
        score = clamp(nods / self.nods_drowsy) if self.nods_drowsy > 0 else 0.0
        active = nods >= self.nods_drowsy
        return CueResult(self.name, self.dimension, score, active, f"うなずき {nods}回")

    def _count(self, times: list[float], pitches: list[float]) -> int:
        # 基準は窓の中央値。窓の平均姿勢が少し下向きでも、そこからの落ち込みだけを見る。
        base = median(pitches)
        enter = base + self.amplitude_deg
        leave = base + self.amplitude_deg / 2  # 出口を低くして、境界の震えで分割されないようにする
        count = 0
        i = 0
        n = len(pitches)
        while i < n:
            if pitches[i] < enter:
                i += 1
                continue
            j = i
            while j < n and pitches[j] >= leave:
                j += 1
            if j < n and times[j] - times[i] <= self.max_seconds:
                count += 1  # 戻ってきた＝うなずき。戻らなければ居眠り姿勢なので数えない
            i = j + 1
        return count
