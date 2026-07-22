"""「本人の安静時」を履歴から推定する共通部品。

生理指標は個人差が大きく、絶対値のしきい値では判定できない。顔面筋電の分野では、
被験者ごとに正規化して安静状態との差分で見るのが標準的な作法になっている。心拍でも
事情は同じなので、その仕組みをここに一本化して各 cue から使う。

実装で守っていること（いずれも実測で踏んだ失敗の対策）:
- 基準と現在値は同じ統計量（中央値）で取る。片方を下位パーセンタイルにすると、変化が
  無くても推定のばらつきぶんだけ差が正に出る。
- 満点までの幅はばらつき(MAD)でも決める。推定が粗いときほど大きな差を要求する。
- 上振れしている間は基準を更新しない。更新し続けると上がった状態が新しい「安静」になり、
  基準の窓より長い負荷を見失う。
- 積むのは一定間隔ごと。毎フレーム積むと直近の1点が窓の長さぶん重複して基準を乗っ取る。
- 更新を止めている間に時間が過ぎても、最低限の標本数は必ず残す。全部古いと判定されて
  基準が消えると、再開直後の数点で新しい基準ができてしまう。
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ...geometry import clamp


def mad(values: list[float]) -> float:
    """中央絶対偏差。外れ値に強い散らばりの指標。標本が少なければ 0。"""
    if len(values) < 3:
        return 0.0
    array = np.asarray(values, dtype=float)
    return float(np.median(np.abs(array - np.median(array))))


class AdaptiveBaseline:
    """安静時の値とそのばらつきを、本人の履歴から育てる。"""

    def __init__(
        self,
        seconds: float = 120.0,
        min_samples: int = 30,
        interval: float = 1.0,
        freeze_at: float = 0.3,
        noise_k: float = 6.0,
        coverage: float = 0.6,
    ) -> None:
        self.seconds = seconds  # 基準に使う履歴の長さ
        self.min_samples = min_samples  # 確定に要る標本数
        self.interval = interval  # 積む間隔（秒）
        self.freeze_at = freeze_at  # このスコアを超えている間は更新しない
        self.noise_k = noise_k  # ばらつき(MAD)の何倍を満点の幅とみなすか
        self.coverage = coverage  # 確定に要る、履歴長に対する被覆率
        self._samples: deque[tuple[float, float]] = deque()
        self._last_at: float | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._last_at = None

    @property
    def required_span(self) -> float:
        return self.coverage * self.seconds

    def update(self, now: float, value: float, ready: bool, score: float) -> None:
        if ready and score >= self.freeze_at:
            return
        if self._last_at is not None and now - self._last_at < self.interval:
            return
        self._last_at = now
        self._samples.append((now, value))
        while len(self._samples) > self.min_samples and self._samples[0][0] < now - self.seconds:
            self._samples.popleft()

    def read(self) -> tuple[float, float, bool]:
        """(基準値, ばらつき, 確定か)。確定＝十分な長さと数の安静を見た状態。"""
        if len(self._samples) < self.min_samples or self._covered() < self.required_span:
            return 0.0, 0.0, False
        values = [v for _, v in self._samples]
        return float(np.median(values)), mad(values), True

    def progress(self) -> float:
        """確立の進行度(0..1)。時間と標本数の両方が揃って初めて 1。"""
        if len(self._samples) < 2:
            return 0.0
        by_time = clamp(self._covered() / self.required_span) if self.required_span > 0 else 1.0
        return min(by_time, clamp(len(self._samples) / max(1, self.min_samples)))

    def score(self, rise: float, span: float, spread: float) -> float:
        """基準からの差を 0..1 にする。満点までの幅は span とばらつきの大きい方。"""
        scale = max(span, self.noise_k * spread)
        return clamp(rise / scale) if scale > 0 else 0.0

    def _covered(self) -> float:
        return self._samples[-1][0] - self._samples[0][0] if len(self._samples) > 1 else 0.0
