"""座標計算ヘルパのテスト。"""

from __future__ import annotations

from alertness.geometry import clamp


def test_clamp_keeps_values_inside_the_range():
    assert clamp(-0.5) == 0.0
    assert clamp(0.4) == 0.4
    assert clamp(3.0) == 1.0
    assert clamp(15.0, 10.0, 20.0) == 15.0


def test_clamp_maps_nan_to_the_lower_bound():
    # 素の max/min だと NaN が上限 1.0 になり、測れなかった値が満点の警告になる。
    assert clamp(float("nan")) == 0.0
    assert clamp(float("nan"), 0.2, 0.9) == 0.2
