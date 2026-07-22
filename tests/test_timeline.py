"""推移の帯（DimensionTimeline）のテスト。"""

from __future__ import annotations

import numpy as np

from alertness.contracts import Assessment, CueResult, Dimension, Level
from alertness.feedback.timeline import DimensionTimeline


def _assessment(level: Level, t: float, valid: bool = True) -> Assessment:
    dim = Dimension("stress", float(level) / 3.0, level)
    cue = CueResult("hr_elevation", "stress", dim.score, level >= Level.MEDIUM, "", None, valid)
    return Assessment(dimensions={"stress": dim}, timestamp=t, cues=(cue,))


def _canvas() -> np.ndarray:
    return np.zeros((240, 400, 3), dtype=np.uint8)


def _colors(img: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(int(c) for c in px) for row in img for px in row}


def test_timeline_draws_levels_over_time():
    tl = DimensionTimeline("stress", span_seconds=10.0, width=100)
    img = _canvas()
    for i in range(50):
        img = _canvas()
        tl.render(img, _assessment(Level.HIGH if i > 25 else Level.NONE, i * 0.2))
    colors = _colors(img)
    assert (0, 0, 255) in colors  # 赤（HIGH）が帯に出ている
    assert (0, 180, 0) in colors  # 緑（NONE）も残っている


def test_timeline_marks_unmeasured_span():
    tl = DimensionTimeline("stress", span_seconds=10.0, width=100)
    img = _canvas()
    for i in range(50):
        img = _canvas()
        tl.render(img, _assessment(Level.MEDIUM, i * 0.2, valid=i > 25))
    assert (90, 90, 90) in _colors(img)  # 計測できていない区間が灰で残る


def test_timeline_drops_samples_outside_span():
    tl = DimensionTimeline("stress", span_seconds=2.0, width=50)
    for i in range(100):
        tl.render(_canvas(), _assessment(Level.NONE, i * 0.1))
    assert len(tl._samples) <= 21  # 2秒ぶん（0.1秒刻み）しか残らない


def test_timeline_ignores_unknown_dimension():
    tl = DimensionTimeline("nope", span_seconds=10.0)
    img = _canvas()
    tl.render(img, _assessment(Level.HIGH, 0.0))
    assert img.sum() == 0  # 何も描かない
