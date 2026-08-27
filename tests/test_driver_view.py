"""運転者向け画面のテスト。

実車では運転者はモニタを見ないので、この画面は「警告が無いときに何も出さない」ことと
「警告のときに大きく出る」ことの2つが要件になる。
"""

from __future__ import annotations

import numpy as np

from alertness.contracts import Assessment, Dimension, Level
from alertness.feedback import driver_view

SIZE = (320, 240)


def _assessment(*dims: Dimension) -> Assessment:
    return Assessment(dimensions={d.name: d for d in dims}, timestamp=0.0)


def _dim(name: str, level: Level, alarm: float = 0.9) -> Dimension:
    return Dimension(name, alarm, level)


def _lit_ratio(image: np.ndarray) -> float:
    """真っ黒でない画素の割合。画面をどれだけ使っているかの目安。"""
    return float(np.mean(image.max(axis=2) > 8))


def test_a_calm_screen_stays_almost_black():
    # 夜間に明るい画面を正面に置くのは、それ自体が危険。
    image = driver_view.render(_assessment(_dim("drowsiness", Level.NONE, 0.1)), SIZE)
    assert _lit_ratio(image) < 0.01


def test_a_low_level_does_not_light_the_screen():
    image = driver_view.render(_assessment(_dim("drowsiness", Level.LOW, 0.4)), SIZE)
    assert _lit_ratio(image) < 0.01


def test_a_warning_uses_the_screen():
    image = driver_view.render(_assessment(_dim("drowsiness", Level.HIGH)), SIZE)
    assert _lit_ratio(image) > 0.05


def test_the_worst_axis_is_the_one_shown():
    calm = _dim("stress", Level.MEDIUM, 0.65)
    urgent = _dim("drowsiness", Level.HIGH, 0.95)
    high = driver_view.render(_assessment(calm, urgent), SIZE)
    only_medium = driver_view.render(_assessment(calm), SIZE)
    # HIGH のときだけ赤（BGR の R が最大）になる。
    assert high[:, :, 2].max() > high[:, :, 1].max()
    assert only_medium[:, :, 1].max() > 0  # MEDIUM は橙なので緑成分が残る


def test_the_canvas_matches_the_requested_size():
    image = driver_view.render(_assessment(_dim("drowsiness", Level.HIGH)), (640, 360))
    assert image.shape == (360, 640, 3)


def test_an_empty_assessment_is_calm():
    image = driver_view.render(_assessment(), SIZE)
    assert _lit_ratio(image) < 0.01
