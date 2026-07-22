"""評価軸の推移を左下に帯で出す。「いつ上がったか」を見るための表示。

ストレスのような遅くて不確かな量は、その瞬間の数字より推移の方が本質を表す。さらに、
上の帯の下に「計測できていたか」の帯を重ねる。これが無いと、本当に上がったのか信号が
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
_INVALID = (90, 90, 90)  # 計測できていない区間
_EMPTY = (45, 45, 45)  # まだ何も無い区間


def _text(img: np.ndarray, s: str, org: tuple[int, int], scale: float, color: tuple) -> None:
    cv2.putText(img, s, org, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, _FONT, scale, color, 1, cv2.LINE_AA)


class DimensionTimeline:
    """1本の評価軸について、直近 span 秒の警告の強さと計測可否を帯で描く。"""

    def __init__(
        self,
        name: str,
        span_seconds: float = 300.0,
        width: int = 300,
        height: int = 18,
        lane_height: int = 6,
    ) -> None:
        self.name = name
        self.span_seconds = span_seconds
        self.width = width
        self.height = height
        self.lane_height = lane_height
        self._samples: deque[tuple[float, Level, bool]] = deque()

    def render(self, img: np.ndarray, assessment: Assessment) -> None:
        dim = assessment.dimensions.get(self.name)
        if dim is None:
            return
        now = assessment.timestamp
        self._append(now, dim.level, self._measured(assessment))

        h = img.shape[0]
        x, y = 16, h - (self.height + self.lane_height + 34)
        _text(
            img,
            f"{dim.display_name}  last {self.span_seconds / 60:.0f}min",
            (x, y),
            0.45,
            (255, 255, 255),
        )
        top = y + 8
        self._draw_columns(img, x, top, now)
        cv2.rectangle(
            img,
            (x, top),
            (x + self.width, top + self.height + self.lane_height),
            (200, 200, 200),
            1,
        )
        _text(
            img,
            "-" + f"{self.span_seconds / 60:.0f}min",
            (x, top + self.height + self.lane_height + 14),
            0.4,
            (170, 170, 170),
        )
        _text(
            img,
            "now",
            (x + self.width - 26, top + self.height + self.lane_height + 14),
            0.4,
            (170, 170, 170),
        )

    def _append(self, now: float, level: Level, measured: bool) -> None:
        self._samples.append((now, level, measured))
        cutoff = now - self.span_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def _measured(self, assessment: Assessment) -> bool:
        # その軸に効く cue が1つでも計測できていれば「計測できていた」とみなす。
        cues = [c for c in assessment.cues if c.dimension == self.name]
        return any(c.valid for c in cues) if cues else True

    def _draw_columns(self, img: np.ndarray, x: int, top: int, now: float) -> None:
        # 1列＝1時間バケツ。同じ列に複数入るときは最も重いレベルを残す（見逃さないため）。
        levels: list[Level | None] = [None] * self.width
        measured: list[bool] = [False] * self.width
        start = now - self.span_seconds
        for t, level, ok in self._samples:
            col = int((t - start) / self.span_seconds * self.width)
            col = min(self.width - 1, max(0, col))
            current = levels[col]
            if current is None or level > current:
                levels[col] = level
            measured[col] = measured[col] or ok

        for col in range(self.width):
            level = levels[col]
            color = _EMPTY if level is None else _LEVEL_COLORS[level]
            cv2.line(img, (x + col, top), (x + col, top + self.height - 1), color, 1)
            lane = (0, 150, 0) if measured[col] else (_INVALID if level is not None else _EMPTY)
            cv2.line(
                img,
                (x + col, top + self.height),
                (x + col, top + self.height + self.lane_height - 1),
                lane,
                1,
            )
