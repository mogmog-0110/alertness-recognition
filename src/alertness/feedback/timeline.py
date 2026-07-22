"""評価軸の推移を左下に折れ線（面）で出す。「いつ上がったか」を見るための表示。

ストレスのような遅くて不確かな量は、その瞬間の数字より推移の方が本質を表す。
縦＝警告の強さ、横＝時間、という普通の時系列グラフにしてある。色だけで段階を表すと
高さが読めず、どのくらい上がったのかが分からないため。

計測できていなかった区間は背景を暗く塗る。これが無いと、本当に上がったのか信号が
壊れていただけなのかを後から切り分けられない。判定には影響しない表示専用。
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ..contracts import Assessment, Level

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# 警告の強さ→色（BGR）。overlay の段階色と揃える。
_LEVEL_COLORS = {
    Level.NONE: (0, 180, 0),
    Level.LOW: (0, 200, 200),
    Level.MEDIUM: (0, 140, 255),
    Level.HIGH: (0, 0, 255),
}
_UNMEASURED = (38, 38, 38)  # 計測できていなかった区間の背景
_GRID = (70, 70, 70)


def _text(img: np.ndarray, s: str, org: tuple[int, int], scale: float, color: tuple) -> None:
    cv2.putText(img, s, org, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, _FONT, scale, color, 1, cv2.LINE_AA)


class DimensionTimeline:
    """1本の評価軸について、直近 span 秒の警告の強さを面グラフで描く。"""

    def __init__(
        self,
        name: str,
        span_seconds: float = 300.0,
        width: int = 300,
        height: int = 60,
        alert_at: float = 0.6,
    ) -> None:
        self.name = name
        self.span_seconds = span_seconds
        self.width = width
        self.height = height
        self.alert_at = alert_at  # ここを超えると警告（levels.medium と揃える）
        self._samples: deque[tuple[float, float, Level, bool]] = deque()

    def render(self, img: np.ndarray, assessment: Assessment) -> None:
        dim = assessment.dimensions.get(self.name)
        if dim is None:
            return
        now = assessment.timestamp
        self._append(now, dim.alarm, dim.level, self._measured(assessment))

        x, top = 16, img.shape[0] - (self.height + 34)
        _text(img, dim.display_name, (x, top - 6), 0.5, (255, 255, 255))
        _text(
            img,
            f"now {dim.level.name} {dim.alarm:.2f}",
            (x + self.width - 118, top - 6),
            0.45,
            _LEVEL_COLORS[dim.level],
        )
        self._draw_plot(img, x, top, now)

    def _append(self, now: float, alarm: float, level: Level, measured: bool) -> None:
        self._samples.append((now, alarm, level, measured))
        cutoff = now - self.span_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def _measured(self, assessment: Assessment) -> bool:
        # その軸に効く cue が1つでも計測できていれば「計測できていた」とみなす。
        cues = [c for c in assessment.cues if c.dimension == self.name]
        return any(c.valid for c in cues) if cues else True

    def _columns(self, now: float) -> tuple[list[float | None], list[Level], list[bool]]:
        # 1列＝1時間バケツ。同じ列に複数入るときは最も重い値を残す（見逃さないため）。
        alarms: list[float | None] = [None] * self.width
        levels: list[Level] = [Level.NONE] * self.width
        measured: list[bool] = [False] * self.width
        start = now - self.span_seconds
        for t, alarm, level, ok in self._samples:
            col = min(self.width - 1, max(0, int((t - start) / self.span_seconds * self.width)))
            current = alarms[col]
            if current is None or alarm > current:
                alarms[col] = alarm
                levels[col] = level
            measured[col] = measured[col] or ok

        # 標本が列より疎なとき（短いセッションや低フレームレート）は隙間が空いて櫛状に
        # 見えてしまうので、直前の値で埋めて連続した線にする。データが始まる前は空のまま。
        last: int | None = None
        for col in range(self.width):
            if alarms[col] is not None:
                last = col
            elif last is not None:
                alarms[col] = alarms[last]
                levels[col] = levels[last]
                measured[col] = measured[last]
        return alarms, levels, measured

    def _draw_plot(self, img: np.ndarray, x: int, top: int, now: float) -> None:
        # 枠線と目盛りの内側を作画領域にする。値0のとき枠線に隠れて見えなくなるのを防ぐ。
        bottom = top + self.height - 1
        inner_top, inner_bottom = top + 1, bottom - 1
        inner_x, inner_w = x + 1, self.width - 1
        span = inner_bottom - inner_top
        cv2.rectangle(img, (x, top), (x + self.width, bottom), (25, 25, 25), -1)

        alarms, levels, measured = self._columns(now)
        for col in range(inner_w):
            if alarms[col] is not None and not measured[col]:
                cv2.line(
                    img, (inner_x + col, inner_top), (inner_x + col, inner_bottom), _UNMEASURED, 1
                )

        # 警告が出るしきい値。この線を越えている区間が「鳴った区間」。
        alert_y = inner_bottom - int(span * self.alert_at)
        for dash in range(inner_x, inner_x + inner_w, 6):
            cv2.line(img, (dash, alert_y), (min(dash + 3, inner_x + inner_w), alert_y), _GRID, 1)

        for col in range(inner_w):
            alarm = alarms[col]
            if alarm is None:
                continue
            value_y = inner_bottom - int(span * min(1.0, max(0.0, alarm)))
            cv2.line(
                img,
                (inner_x + col, value_y),
                (inner_x + col, inner_bottom),
                _LEVEL_COLORS[levels[col]],
                1,
            )

        cv2.rectangle(img, (x, top), (x + self.width, bottom), (200, 200, 200), 1)
        _text(img, f"-{self.span_seconds / 60:.0f}min", (x, bottom + 14), 0.4, (170, 170, 170))
        _text(img, "alert", (x + self.width + 4, alert_y + 4), 0.35, (150, 150, 150))
        _text(img, "now", (x + self.width - 26, bottom + 14), 0.4, (170, 170, 170))
