"""走行中の分布から「前方」を推定する。

起動時数秒の正面キャリブは、カメラを注視すべき方向からずらして取り付けると以降ずっと
その角度ぶんずれる（設定にも「そのとき顔が少し傾いていると以降ずっと十数度ずれる」と
書いてあるとおり）。運転者は時間の大半を前方に向けているので、走行中の分布の最頻値が
前方になる。取り付け位置を測らなくても、走っているうちに基準が定まる。

中央値ではなく最頻値を使う。片側のミラーをよく見る癖や助手席と話す時間があると、
中央値はそちらへ引かれる。最頻値なら「一番長く留まっていた向き」が残る。

前方から外れている間も含めて全部積む。外れた時間を除くと、いま前方だと思っている向きの
標本だけが集まり、最初のずれがそのまま固定される（自己強化）。分布全体を見るからこそ
「一番長く見ている向き」が浮かび上がる。
"""

from __future__ import annotations

import math
from collections import Counter, deque


class ForwardBaseline:
    """1つの量（視線ズレ・yaw・pitch のどれか）について、前方に当たる値を推定する。"""

    def __init__(
        self,
        bin_width: float,
        seconds: float = 180.0,
        min_samples: int = 60,
        interval: float = 0.5,
        min_share: float = 0.2,
    ) -> None:
        self.bin_width = bin_width  # 最頻値を探す刻み幅
        self.seconds = seconds  # 分布を見る履歴の長さ
        self.min_samples = min_samples  # 確定に要る標本数
        self.interval = interval  # 積む間隔（秒）。毎フレーム積むと直近が分布を乗っ取る
        # 最頻の山がこの割合を占めていなければ確定しない。分布が平らなときに最頻値を
        # 名乗ると、たまたま多かっただけの向きを前方に据えてしまう。
        self.min_share = min_share
        self._samples: deque[tuple[float, float]] = deque()
        self._last_at: float | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._last_at = None

    def update(self, now: float, value: float) -> None:
        if math.isnan(value):
            return
        if self._last_at is not None and now - self._last_at < self.interval:
            return
        self._last_at = now
        self._samples.append((now, value))
        while self._samples and self._samples[0][0] < now - self.seconds:
            self._samples.popleft()

    def read(self) -> tuple[float, bool]:
        """(前方に当たる値, 確定か)。確定していなければ (0.0, False)。"""
        if len(self._samples) < self.min_samples or self.bin_width <= 0:
            return 0.0, False
        values = [v for _, v in self._samples]
        counts = Counter(round(v / self.bin_width) for v in values)
        center, hits = counts.most_common(1)[0]
        if hits / len(values) < self.min_share:
            return 0.0, False
        # 山の中身で median を取り直す。ビンの中心をそのまま返すと刻み幅ぶん粗い。
        inside = sorted(v for v in values if abs(round(v / self.bin_width) - center) <= 1)
        return inside[len(inside) // 2], True

    def progress(self) -> float:
        if self.min_samples <= 0:
            return 1.0
        return min(1.0, len(self._samples) / self.min_samples)


class ForwardPose:
    """視線ズレ・yaw・pitch の3つをまとめて推定し、その場の値から差し引く。"""

    def __init__(self, seconds: float = 180.0, min_samples: int = 60, interval: float = 0.5):
        common = {"seconds": seconds, "min_samples": min_samples, "interval": interval}
        self._gaze = ForwardBaseline(bin_width=0.01, **common)
        self._yaw = ForwardBaseline(bin_width=2.0, **common)
        self._pitch = ForwardBaseline(bin_width=2.0, **common)

    def reset(self) -> None:
        for baseline in (self._gaze, self._yaw, self._pitch):
            baseline.reset()

    def update(self, now: float, gaze_dx: float, yaw: float, pitch: float) -> None:
        self._gaze.update(now, gaze_dx)
        self._yaw.update(now, yaw)
        self._pitch.update(now, pitch)

    def correct(self, gaze_dx: float, yaw: float, pitch: float) -> tuple[float, float, float]:
        """推定できた軸だけ基準を差し引く。まだ確定していない軸はそのまま返す。"""
        return (
            _shift(gaze_dx, self._gaze),
            _shift(yaw, self._yaw),
            _shift(pitch, self._pitch),
        )

    def progress(self) -> float:
        return min(b.progress() for b in (self._gaze, self._yaw, self._pitch))

    @property
    def ready(self) -> bool:
        return all(b.read()[1] for b in (self._gaze, self._yaw, self._pitch))


def _shift(value: float, baseline: ForwardBaseline) -> float:
    center, ready = baseline.read()
    if not ready or math.isnan(value):
        return value
    return value - center
