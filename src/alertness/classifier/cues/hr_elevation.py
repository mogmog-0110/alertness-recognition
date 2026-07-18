"""ストレスの手がかり（rPPG使用）。安静時の基準より心拍が上がる＝ストレスの疑い。

映像だけの幾何特徴ではストレスを測れないので、rPPG が出す hr_bpm を手がかりにする。
rPPG が無効／推定前で hr_bpm が無いときは inactive（ストレスを none と断定しない）。
基準心拍(baseline_bpm)は個人差が大きく、ここでは暫定の固定値。実運用では起動時の安静
キャリブで各人の基準を採るべき（暫定実装）。品質の低い推定は捨てる。
"""

from __future__ import annotations

import math

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._support import window_values


class HrElevationCue:
    name = "hr_elevation"
    dimension = "stress"

    def __init__(
        self,
        baseline_bpm: float = 70.0,
        span_bpm: float = 30.0,
        min_quality: float = 0.15,
        sustained_seconds: float = 5.0,
    ) -> None:
        self.baseline_bpm = baseline_bpm  # 安静時の目安（暫定・要キャリブ）
        self.span_bpm = span_bpm  # baseline から満点までの上昇幅
        self.min_quality = min_quality  # これ未満の推定は信用しない
        self.sustained_seconds = sustained_seconds

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし")

        window = max(2.0, self.sustained_seconds)
        times, hrs = window_values(obs, "hr_bpm", window, float("nan"))
        _, quals = window_values(obs, "rppg_quality", window, 0.0)
        valid = [
            h
            for h, q in zip(hrs, quals, strict=False)
            if not math.isnan(h) and q >= self.min_quality
        ]
        if not valid:
            return CueResult(self.name, self.dimension, 0.0, False, "心拍なし")

        hr = valid[-1]  # 直近の有効な推定
        score = clamp((hr - self.baseline_bpm) / self.span_bpm) if self.span_bpm > 0 else 0.0
        active = score >= 0.5
        return CueResult(self.name, self.dimension, score, active, f"HR {hr:.0f}bpm")
