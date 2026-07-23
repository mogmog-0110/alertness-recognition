"""ストレスの手がかり（rPPG使用）。生理的な覚醒（arousal）の上振れ＝ストレスの疑い。

映像だけの幾何特徴では心拍を測れないので rPPG を使う。指標は2系統で、機会的に使い分ける:
- HRV(RMSSD): 良質・安定した窓でだけ rPPG が出す。安静時より下がる＝ストレス。より確からしい。
- 心拍(HR): 常時出る頑健な指標。安静時より上がる＝ストレス。HRV が無いときの退避。

心拍上昇は感度は高いが特異度が低い（体動・姿勢・会話・室温でも上がる）。単独でストレスを
名乗らせず、表情の cue と揃ったときに軸が立つよう、stress 軸は combine: weighted で使う。

安静の基準は AdaptiveBaseline に任せる（表情の cue と同じ仕組み。個人差の扱いと、
基準が上昇を追いかけない工夫はそちらに集約してある）。

測れていないときに 0（＝ストレスなし）を出すと嘘になるので、次の2つを守る:
- 安静基準が未確立の間は判定を出さない。暫定の固定基準で判定すると、平常心拍が高い人は
  起動直後に必ずストレス高と出てしまう。
- 頭が動いている間の rPPG は当てにならないので判定を止め、直前の値を短時間だけ保つ。
  動くたびに 0 へ落ちると「動かすとストレスが下がる」という誤った像を与えるため。
"""

from __future__ import annotations

import math

import numpy as np

from ...contracts import CueResult, Observation
from ._baseline import AdaptiveBaseline
from ._baseline import mad as _mad
from ._support import window_values


class HrElevationCue:
    name = "hr_elevation"
    dimension = "stress"

    def __init__(
        self,
        span_bpm: float = 10.0,
        min_quality: float = 0.6,
        sustained_seconds: float = 5.0,
        baseline_bpm: float = 70.0,
        adaptive_baseline: bool = True,
        baseline_seconds: float = 120.0,
        noise_k: float = 6.0,
        freeze_at: float = 0.3,
        rmssd_span: float = 25.0,
        hrv_min_samples: int = 4,
        require_calibration: bool = True,
        max_motion_deg: float = 6.0,
        hold_seconds: float = 10.0,
        min_rest_samples: int = 30,
        rest_interval: float = 1.0,
    ) -> None:
        self.span_bpm = span_bpm  # HR: baseline から満点までの上昇幅
        self.min_quality = min_quality  # これ未満の HR 推定は信用しない
        self.sustained_seconds = sustained_seconds  # 現在値を取る窓
        self.baseline_bpm = baseline_bpm  # 履歴が浅いときの暫定基準
        self.adaptive_baseline = adaptive_baseline  # 本人の安静時を履歴から推定するか
        self.rmssd_span = rmssd_span  # HRV: baseline からの低下幅で満点
        self.hrv_min_samples = hrv_min_samples  # 安静基準を作るのに要る HRV 標本数
        self.require_calibration = require_calibration  # 安静基準が確立するまで判定を出さない
        self.max_motion_deg = max_motion_deg  # 頭部のふらつきがこれを超えたら測定を止める
        self.hold_seconds = hold_seconds  # 測れない間、直前の値を保つ上限
        self._rest = AdaptiveBaseline(
            seconds=baseline_seconds,
            min_samples=min_rest_samples,
            interval=rest_interval,
            freeze_at=freeze_at,
            noise_k=noise_k,
        )
        self._held = 0.0  # 直前に出せた値
        self._held_at: float | None = None  # その時刻

    def evaluate(self, obs: Observation) -> CueResult:
        progress = self._progress()
        now = obs.features.timestamp
        if not obs.features.face_present:
            self._held, self._held_at = 0.0, None  # 離席したら保持をやめる
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし", progress, False)

        if self._moving(obs):
            return self._hold(now, progress, "頭部が動いている")

        hrv = self._hrv_stress(obs)  # 確度が高い窓なら HRV を優先。
        if hrv is not None:
            score, detail = hrv
            return self._emit(now, score, detail, progress)

        recent = self._valid(obs, "hr_bpm", self.sustained_seconds, self.min_quality)
        if not recent:
            return self._hold(now, progress, "心拍なし")

        current = float(np.median([v for _, v in recent]))
        baseline, spread, ready = self._read_baseline()
        score = self._rest.score(current - baseline, self.span_bpm, spread) if ready else 0.0
        self._rest.update(now, current, ready, score)
        if self.require_calibration and not ready:
            detail = "安静基準を測定中"
            return CueResult(self.name, self.dimension, 0.0, False, detail, progress, False)

        detail = f"HR {current:.0f} base {baseline:.0f} +-{spread:.0f}"
        return self._emit(now, score, detail, progress)

    def _emit(self, now: float, score: float, detail: str, progress: float | None) -> CueResult:
        self._held, self._held_at = score, now
        return CueResult(self.name, self.dimension, score, score >= 0.5, detail, progress)

    def _hold(self, now: float, progress: float | None, reason: str) -> CueResult:
        # 測れない間は直前の値を保つ。保持中は active にしない（根拠として数えない）。
        if self._held_at is None or now - self._held_at > self.hold_seconds:
            return CueResult(self.name, self.dimension, 0.0, False, reason, progress, False)
        detail = f"{reason}（保持）"
        return CueResult(self.name, self.dimension, self._held, False, detail, progress, False)

    def _moving(self, obs: Observation) -> bool:
        # 頭部の振れ幅で動きを見る。動いている間の額の色変化は脈より動きの影響が大きい。
        # 生の pitch/yaw は ±180 を跨いで折り返すので、必ず正規化済み（畳んだ）値を使う。
        # 生値を使うと、静止していても標準偏差が 180 度近くになり常時「動いている」になる。
        if self.max_motion_deg <= 0:
            return False
        window = max(1.0, self.sustained_seconds)
        for key in ("yaw_rel", "pitch_rel"):
            _, values = window_values(obs, key, window, float("nan"))
            clean = [v for v in values if not math.isnan(v)]
            if len(clean) >= 3 and float(np.std(clean)) > self.max_motion_deg:
                return True
        return False

    def _hrv_stress(self, obs: Observation) -> tuple[float, str] | None:
        # HRV(RMSSD)が本人の安静基準を作れるほど貯まり、直近にも値があれば (score, detail)。
        samples = self._valid(obs, "hrv_rmssd", self._rest.seconds)
        if len(samples) < self.hrv_min_samples:
            return None
        recent = self._valid(obs, "hrv_rmssd", self.sustained_seconds * 2)
        if not recent or self.rmssd_span <= 0:
            return None
        current = float(np.median([v for _, v in recent]))
        values = [v for _, v in samples]
        baseline = float(np.median(values))  # 現在値と同じ統計量にする（偏りを作らない）
        spread = _mad(values)
        score = self._rest.score(baseline - current, self.rmssd_span, spread)  # 低RMSSD＝ストレス
        return score, f"HRV {current:.0f}ms base {baseline:.0f} +-{spread:.0f}"

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

    def _progress(self) -> float | None:
        # 安静基準（HR）を確立するキャリブの進行度(0..1)。固定基準なら較正不要で None。
        return self._rest.progress() if self.adaptive_baseline else None

    def _read_baseline(self) -> tuple[float, float, bool]:
        # 返り値は (基準bpm, 推定のばらつき, 確定か)。確定＝本人の安静を十分な長さ見た状態。
        if not self.adaptive_baseline:
            return self.baseline_bpm, 0.0, True
        base, spread, ready = self._rest.read()
        return (base, spread, True) if ready else (self.baseline_bpm, 0.0, False)
