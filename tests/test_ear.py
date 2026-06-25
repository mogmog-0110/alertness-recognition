"""EAR 計算のテスト。"""

from __future__ import annotations

import pytest

from alertness.features.ear import eye_aspect_ratio, is_eye_closed


def test_known_value():
    # 縦の距離が各2、横が2 なので EAR = (2+2)/(2*2) = 1.0
    points = [(0, 0), (0, 1), (0, 1), (2, 0), (0, -1), (0, -1)]
    assert eye_aspect_ratio(points) == pytest.approx(1.0)


def test_open_eye_higher_than_closed():
    open_eye = [(0, 0), (0.5, 1), (1.5, 1), (2, 0), (1.5, -1), (0.5, -1)]
    closed_eye = [(0, 0), (0.5, 0.1), (1.5, 0.1), (2, 0), (1.5, -0.1), (0.5, -0.1)]
    assert eye_aspect_ratio(open_eye) > eye_aspect_ratio(closed_eye)


def test_zero_width_is_safe():
    degenerate = [(0, 0), (0, 1), (0, 1), (0, 0), (0, -1), (0, -1)]
    assert eye_aspect_ratio(degenerate) == 0.0


def test_requires_six_points():
    with pytest.raises(ValueError):
        eye_aspect_ratio([(0, 0), (1, 1)])


def test_is_eye_closed_uses_baseline():
    # 基準0.3の6割=0.18 を下回ると閉眼
    assert is_eye_closed(0.1, open_baseline=0.3, closed_ratio=0.6)
    assert not is_eye_closed(0.25, open_baseline=0.3, closed_ratio=0.6)
