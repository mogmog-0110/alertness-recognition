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


def _bar_height(img: np.ndarray, col: int, top: int, bottom: int) -> int:
    """指定列で、背景でない画素が下から何段積まれているか。"""
    count = 0
    for y in range(bottom, top, -1):
        if tuple(int(c) for c in img[y, col]) in ((25, 25, 25), (38, 38, 38)):
            break
        count += 1
    return count


def test_timeline_draws_levels_over_time():
    tl = DimensionTimeline("stress", span_seconds=10.0, width=100)
    img = _canvas()
    for i in range(50):
        img = _canvas()
        tl.render(img, _assessment(Level.HIGH if i > 25 else Level.NONE, i * 0.2))
    colors = _colors(img)
    assert (0, 0, 255) in colors  # 赤（HIGH）が出ている
    assert (0, 180, 0) in colors  # 緑（NONE）も残っている


def test_timeline_encodes_magnitude_as_height():
    # 色だけでなく高さで大きさが読めること（見方が分かるための肝）。
    tl = DimensionTimeline("stress", span_seconds=10.0, width=100, height=60)
    img = _canvas()
    for i in range(25):
        img = _canvas()
        tl.render(img, _assessment(Level.NONE, i * 0.2))  # alarm 0.0
    low_h = _bar_height(img, 16 + 60, 240 - 94, 240 - 36)
    for i in range(25, 50):
        img = _canvas()
        tl.render(img, _assessment(Level.HIGH, i * 0.2))  # alarm 1.0
    high_h = _bar_height(img, 16 + 60, 240 - 94, 240 - 36)
    assert high_h > low_h + 30  # 値が大きいほど棒が高い


def test_timeline_marks_unmeasured_span():
    tl = DimensionTimeline("stress", span_seconds=10.0, width=100)
    img = _canvas()
    for i in range(50):
        img = _canvas()
        tl.render(img, _assessment(Level.MEDIUM, i * 0.2, valid=i > 25))
    assert (38, 38, 38) in _colors(img)  # 計測できていない区間の背景が暗く塗られる


def test_timeline_fills_gaps_between_sparse_samples():
    # 標本が列より疎でも、隙間が空いて櫛状にならないこと。
    tl = DimensionTimeline("stress", span_seconds=10.0, width=200, height=60)
    img = _canvas()
    for i in range(20):  # 20標本を200列に載せる
        img = _canvas()
        tl.render(img, _assessment(Level.HIGH, i * 0.5))
    alarms, _, _ = tl._columns(tl._samples[-1][0])
    filled = [a for a in alarms if a is not None]
    assert len(filled) > 150  # 直前の値で埋まっている


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
