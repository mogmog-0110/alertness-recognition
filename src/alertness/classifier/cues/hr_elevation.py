"""ストレスの手がかり（rPPG使用）。安静時より心拍が上がる＝ストレスの疑い。

映像だけの幾何特徴ではストレスを測れないので、rPPG が出す hr_bpm を手がかりにする。
基準心拍は個人差が大きく固定値では外しやすいため、既定では履歴から本人の安静時心拍を
推定して相対化する（adaptive）。十分な履歴が貯まるまでは固定の baseline_bpm で代用する。
品質(rppg_quality)の低い推定は捨てる。rPPG が無効／推定前で hr_bpm が無いときは inactive
（ストレスを none と断定しない）。
"""

from __future__ import annotations

import math

import numpy as np

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._support import window_values


class HrElevationCue:
    name = "hr_elevation"
    dimension = "stress"

    def __init__(
        self,
        span_bpm: float = 25.0,
        min_quality: float = 0.1,
        sustained_seconds: float = 5.0,
        baseline_bpm: float = 70.0,
        adaptive_baseline: bool = True,
        baseline_seconds: float = 45.0,
        baseline_percentile: float = 25.0,
    ) -> None:
        self.span_bpm = span_bpm  # baseline から満点までの上昇幅
        self.min_quality = min_quality  # これ未満の推定は信用しない
        self.sustained_seconds = sustained_seconds
        self.baseline_bpm = baseline_bpm  # 履歴が浅いときの暫定基準
        self.adaptive_baseline = adaptive_baseline  # 本人の安静時心拍を履歴から推定するか
        self.baseline_seconds = baseline_seconds  # 基準を測る履歴の長さ
        self.baseline_percentile = baseline_percentile  # 基準に使う下側パーセンタイル

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし")

        recent = self._valid_hr(obs, self.sustained_seconds)
        if not recent:
            return CueResult(self.name, self.dimension, 0.0, False, "心拍なし")

        current = float(np.median([h for _, h in recent]))
        baseline = self._baseline(obs)
        score = clamp((current - baseline) / self.span_bpm) if self.span_bpm > 0 else 0.0
        active = score >= 0.5
        detail = f"HR {current:.0f}/base {baseline:.0f}"
        return CueResult(self.name, self.dimension, score, active, detail)

    def _valid_hr(self, obs: Observation, seconds: float) -> list[tuple[float, float]]:
        # 品質が足りる有効な (時刻, hr) だけを返す。
        window = max(2.0, seconds)
        times, hrs = window_values(obs, "hr_bpm", window, float("nan"))
        _, quals = window_values(obs, "rppg_quality", window, 0.0)
        return [
            (t, h)
            for t, h, q in zip(times, hrs, quals, strict=False)
            if not math.isnan(h) and q >= self.min_quality
        ]

    def _baseline(self, obs: Observation) -> float:
        if not self.adaptive_baseline:
            return self.baseline_bpm
        samples = self._valid_hr(obs, self.baseline_seconds)
        # 履歴が baseline_seconds の6割以上を覆っていれば、本人の下側心拍を基準にする。
        if samples and (samples[-1][0] - samples[0][0]) >= 0.6 * self.baseline_seconds:
            return float(np.percentile([h for _, h in samples], self.baseline_percentile))
        return self.baseline_bpm
