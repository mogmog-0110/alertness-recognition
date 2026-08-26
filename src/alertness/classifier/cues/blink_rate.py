"""ストレスの手がかり（瞬きの頻度）。安静時より瞬きが増える＝緊張・認知負荷。

瞬きの回数は心理的ストレスと認知負荷で増えることが報告されている。カメラだけで取れて
追加のハードが要らず、閉眼エピソードの切り出しは眠気側の cue と同じ部品を使い回せる。

眠気側の blink_dynamics が「1回の瞬きの長さ」を見るのに対し、こちらは「単位時間あたりの
回数」を見る。眠気では回数が減って1回が長くなり、ストレスでは回数が増えて1回は短いままに
なるので、2つを別の軸に置いても取り合いにはならない。

安静時の瞬き回数は 10〜20 回/分と個人差が大きいので、本人の安静との差で見る。
単独ではストレスを断定しない（active を立てない）。乾燥・エアコンの風・画面の見過ぎ
でも増えるため、特異度が足りない。
"""

from __future__ import annotations

from ...contracts import CueResult, Observation
from ._baseline import AdaptiveBaseline
from ._episodes import closure_episodes
from ._support import window_coverage, window_values


class BlinkRateCue:
    name = "blink_rate"
    dimension = "stress"

    def __init__(
        self,
        window_seconds: float = 60.0,
        closed_ratio: float = 0.6,
        open_ratio: float = 0.7,
        max_blink_seconds: float = 0.5,
        span_rate: float = 8.0,
        baseline_seconds: float = 180.0,
        min_rest_samples: int = 20,
        rest_interval: float = 5.0,
        freeze_at: float = 0.3,
        noise_k: float = 6.0,
        max_yaw: float = 25.0,
        min_coverage: float = 0.6,
    ) -> None:
        self.window_seconds = window_seconds  # 回数を数える窓
        self.closed_ratio = closed_ratio
        self.open_ratio = open_ratio
        # 長い閉眼は瞬きではなく眠気。数に混ぜると、眠くなるほどストレスが上がって見える。
        self.max_blink_seconds = max_blink_seconds
        self.span_rate = span_rate  # 基準からこれだけ増えると満点（回/分）
        self.max_yaw = max_yaw  # 横顔では EAR が壊れるので数えない
        self.min_coverage = min_coverage  # 窓のうち顔が見えていた時間の下限
        self._rest = AdaptiveBaseline(
            seconds=baseline_seconds,
            min_samples=min_rest_samples,
            interval=rest_interval,
            freeze_at=freeze_at,
            noise_k=noise_k,
        )

    def reset(self) -> None:
        """安静基準を捨てる。安静時の瞬き回数は 10〜20 回/分と個人差が大きい。"""
        self._rest.reset()

    def evaluate(self, obs: Observation) -> CueResult:
        progress = self._rest.progress()
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし", progress, False)
        if abs(obs.features.get("yaw_rel", 0.0)) > self.max_yaw:
            return CueResult(self.name, self.dimension, 0.0, False, "横向き", progress, False)

        coverage = window_coverage(obs, self.window_seconds)
        if coverage < self.min_coverage:
            # 数えた回数を「窓の長さ」で割ると、見えていなかった時間のぶん頻度が下がる。
            detail = f"計測不足 {coverage:.0%}"
            return CueResult(self.name, self.dimension, 0.0, False, detail, progress, False)

        rate = self._rate(obs)
        if rate is None:
            return CueResult(self.name, self.dimension, 0.0, False, "観察中", progress, False)

        base, spread, ready = self._rest.read()
        score = self._rest.score(rate - base, self.span_rate, spread) if ready else 0.0
        self._rest.update(obs.features.timestamp, rate, ready, score)
        if not ready:
            detail = "瞬き頻度の基準を測定中"
            return CueResult(self.name, self.dimension, 0.0, False, detail, progress, False)

        detail = f"瞬き {rate:.0f}/分 base {base:.0f} +-{spread:.1f}"
        # active は立てない。乾燥やエアコンの風でも増えるので、単独では断定できない。
        return CueResult(self.name, self.dimension, score, False, detail, progress, True)

    def _rate(self, obs: Observation) -> float | None:
        """窓内の瞬き回数を「回/分」に直す。窓が短すぎれば None。"""
        times, ears = window_values(obs, "ear_norm", self.window_seconds, 1.0)
        if len(times) < 2:
            return None
        span = times[-1] - times[0]
        if span < self.window_seconds * 0.5:
            return None
        episodes = closure_episodes(times, ears, self.closed_ratio, self.open_ratio)
        blinks = [e for e in episodes if e.duration <= self.max_blink_seconds]
        return len(blinks) * 60.0 / span
