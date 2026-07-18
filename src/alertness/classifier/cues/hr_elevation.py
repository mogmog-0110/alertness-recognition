"""ストレスの手がかり（rPPG使用）。生理的な覚醒（arousal）の上振れ＝ストレスの疑い。

映像だけの幾何特徴ではストレスを測れないので rPPG を使う。指標は2系統で、機会的に使い分ける:
- HRV(RMSSD): 良質・安定した窓でだけ rPPG が出す。安静時より下がる＝ストレス。より確からしい。
- 心拍(HR): 常時出る頑健な指標。安静時より上がる＝ストレス。HRV が無いときの退避。

どちらも「本人の安静時」を履歴から推定して相対化する（adaptive）。HRV が確度高く得られていれば
そちらを優先し、無ければ HR に退避する。rPPG 無効／推定前で何も無いときは inactive
（ストレスを none と断定しない）。キャリブ進行度は常時出る HR 基準の確立度で表す。
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
        min_quality: float = 0.05,
        sustained_seconds: float = 5.0,
        baseline_bpm: float = 70.0,
        adaptive_baseline: bool = True,
        baseline_seconds: float = 45.0,
        baseline_percentile: float = 25.0,
        rmssd_span: float = 25.0,
        rmssd_percentile: float = 75.0,
        hrv_min_samples: int = 4,
    ) -> None:
        self.span_bpm = span_bpm  # HR: baseline から満点までの上昇幅
        self.min_quality = min_quality  # これ未満の HR 推定は信用しない
        self.sustained_seconds = sustained_seconds
        self.baseline_bpm = baseline_bpm  # 履歴が浅いときの暫定基準
        self.adaptive_baseline = adaptive_baseline  # 本人の安静時を履歴から推定するか
        self.baseline_seconds = baseline_seconds  # 基準を測る履歴の長さ
        self.baseline_percentile = baseline_percentile  # HR 基準に使う下側パーセンタイル
        self.rmssd_span = rmssd_span  # HRV: baseline からの低下幅で満点
        self.rmssd_percentile = rmssd_percentile  # HRV 基準に使う上側パーセンタイル（安静=高RMSSD）
        self.hrv_min_samples = hrv_min_samples  # 安静基準を作るのに要る HRV 標本数

    def evaluate(self, obs: Observation) -> CueResult:
        progress = self._progress(obs)
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし", progress)

        hrv = self._hrv_stress(obs)  # 確度が高い窓なら HRV を優先。
        if hrv is not None:
            score, detail = hrv
            return CueResult(self.name, self.dimension, score, score >= 0.5, detail, progress)

        recent = self._valid(obs, "hr_bpm", self.sustained_seconds, self.min_quality)
        if not recent:
            return CueResult(self.name, self.dimension, 0.0, False, "心拍なし", progress)

        current = float(np.median([v for _, v in recent]))
        baseline, ready = self._baseline(obs)
        score = clamp((current - baseline) / self.span_bpm) if self.span_bpm > 0 else 0.0
        detail = f"HR {current:.0f} base {baseline:.0f}{'' if ready else ' (warm)'}"
        return CueResult(self.name, self.dimension, score, score >= 0.5, detail, progress)

    def _hrv_stress(self, obs: Observation) -> tuple[float, str] | None:
        # HRV(RMSSD)が本人の安静基準を作れるほど貯まり、直近にも値があれば (score, detail)。
        samples = self._valid(obs, "hrv_rmssd", self.baseline_seconds)
        if len(samples) < self.hrv_min_samples:
            return None
        recent = self._valid(obs, "hrv_rmssd", self.sustained_seconds * 2)
        if not recent or self.rmssd_span <= 0:
            return None
        current = float(np.median([v for _, v in recent]))
        baseline = float(np.percentile([v for _, v in samples], self.rmssd_percentile))
        score = clamp((baseline - current) / self.rmssd_span)  # 安静より低RMSSD＝ストレス
        return score, f"HRV {current:.0f}ms base {baseline:.0f}"

    def _valid(
        self, obs: Observation, key: str, seconds: float, quality: float | None = None
    ) -> list[tuple[float, float]]:
        # 有効な (時刻, 値) を返す。quality 指定時は rppg_quality で足切りする。
        window = max(2.0, seconds)
        times, vals = window_values(obs, key, window, float("nan"))
        if quality is None:
            return [(t, v) for t, v in zip(times, vals, strict=False) if not math.isnan(v)]
        _, quals = window_values(obs, "rppg_quality", window, 0.0)
        return [
            (t, v)
            for t, v, q in zip(times, vals, quals, strict=False)
            if not math.isnan(v) and q >= quality
        ]

    def _progress(self, obs: Observation) -> float | None:
        # 安静基準（HR）を確立するキャリブの進行度(0..1)。固定基準なら較正不要で None。
        if not self.adaptive_baseline:
            return None
        samples = self._valid(obs, "hr_bpm", self.baseline_seconds, self.min_quality)
        if len(samples) < 2:
            return 0.0
        covered = samples[-1][0] - samples[0][0]
        required = 0.6 * self.baseline_seconds  # _baseline の確定条件と揃える
        return clamp(covered / required) if required > 0 else 1.0

    def _baseline(self, obs: Observation) -> tuple[float, bool]:
        # 返り値は (基準bpm, 確定か)。確定＝本人の履歴から推定できた状態。
        if not self.adaptive_baseline:
            return self.baseline_bpm, True
        samples = self._valid(obs, "hr_bpm", self.baseline_seconds, self.min_quality)
        # 履歴が baseline_seconds の6割以上を覆っていれば、本人の下側心拍を基準にする。
        if samples and (samples[-1][0] - samples[0][0]) >= 0.6 * self.baseline_seconds:
            return float(np.percentile([v for _, v in samples], self.baseline_percentile)), True
        return self.baseline_bpm, False
