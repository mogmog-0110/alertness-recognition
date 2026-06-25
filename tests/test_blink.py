"""瞬き関連の時系列計算のテスト。"""

from __future__ import annotations

import pytest

from alertness.features.blink import blink_rate_per_minute, current_closed_duration


def test_trailing_closed_duration():
    times = [0.0, 0.1, 0.2, 0.3]
    closed = [False, True, True, True]
    assert current_closed_duration(times, closed) == pytest.approx(0.2)


def test_no_duration_when_currently_open():
    assert current_closed_duration([0.0, 1.0, 2.0], [True, True, False]) == 0.0


def test_blink_rate_counts_falling_edges():
    times = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    closed = [False, True, False, True, False, False, False]
    # 開→閉の遷移が2回、経過60秒 → 毎分2回
    assert blink_rate_per_minute(times, closed) == pytest.approx(2.0)


def test_blink_rate_needs_two_points():
    assert blink_rate_per_minute([0.0], [True]) == 0.0
