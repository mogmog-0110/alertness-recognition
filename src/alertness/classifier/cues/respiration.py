"""ストレスの手がかり（呼吸）。安静時より呼吸が速くなる＝交感神経の亢進。

心拍と同じ交感神経の亢進を、別の経路で見る指標。心拍より雑音に強いという利点がある。
心拍の推定は脈という小さな成分を額の色から取り出すので照明と体動に弱いが、呼吸は
胸郭の動きが顔全体を周期的に動かすぶん、信号が大きい。追加のハードは要らず、
rPPG がすでに集めている肌色の時系列の低周波成分から取れる。

個人差が大きい（安静時 12〜20 回/分）ので、心拍・表情と同じく本人の安静との差で見る。
単独ではストレスを断定しない（active を立てない）。呼吸数は会話・あくび・咳でも上がる
うえ、rPPG の低周波帯は照明のちらつきとも重なるため、特異度が足りない。
"""

from __future__ import annotations

import math

import numpy as np

from ...contracts import CueResult, Observation
from ._baseline import AdaptiveBaseline
from ._support import window_values


class RespirationCue:
    name = "respiration"
    dimension = "stress"

    def __init__(
        self,
        span_rpm: float = 4.0,
        min_quality: float = 0.4,
        sustained_seconds: float = 15.0,
        baseline_seconds: float = 180.0,
        min_rest_samples: int = 20,
        rest_interval: float = 2.0,
        freeze_at: float = 0.3,
        noise_k: float = 6.0,
    ) -> None:
        self.span_rpm = span_rpm  # 基準からこれだけ上がると満点（回/分）
        self.min_quality = min_quality  # これ未満の推定は基準にも現在値にも使わない
        self.sustained_seconds = sustained_seconds  # 現在値を取る窓
        self._rest = AdaptiveBaseline(
            seconds=baseline_seconds,
            min_samples=min_rest_samples,
            interval=rest_interval,
            freeze_at=freeze_at,
            noise_k=noise_k,
        )

    def reset(self) -> None:
        """安静基準を捨てる。安静時呼吸数は 12〜20 回/分と個人差が大きい。"""
        self._rest.reset()

    def evaluate(self, obs: Observation) -> CueResult:
        progress = self._rest.progress()
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし", progress, False)

        recent = self._usable(obs, self.sustained_seconds)
        if not recent:
            detail = "呼吸なし"  # rPPG 無効か、窓が埋まっていない。ストレス none と断定しない
            return CueResult(self.name, self.dimension, 0.0, False, detail, progress, False)

        current = float(np.median(recent))
        base, spread, ready = self._rest.read()
        score = self._rest.score(current - base, self.span_rpm, spread) if ready else 0.0
        self._rest.update(obs.features.timestamp, current, ready, score)
        if not ready:
            detail = "呼吸の基準を測定中"
            return CueResult(self.name, self.dimension, 0.0, False, detail, progress, False)

        detail = f"呼吸 {current:.0f}/分 base {base:.0f} +-{spread:.1f}"
        # active は立てない。会話・あくび・照明のちらつきでも動くので、単独では断定できない。
        return CueResult(self.name, self.dimension, score, False, detail, progress, True)

    def _usable(self, obs: Observation, seconds: float) -> list[float]:
        window = max(2.0, seconds)
        _, values = window_values(obs, "resp_rpm", window, float("nan"))
        _, quals = window_values(obs, "resp_quality", window, 0.0)
        return [
            v
            for v, q in zip(values, quals, strict=False)
            if not math.isnan(v) and q >= self.min_quality
        ]
