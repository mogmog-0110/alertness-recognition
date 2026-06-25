"""PERCLOS 計算のテスト。"""

from __future__ import annotations

from alertness.features.perclos import perclos


def test_empty_is_zero():
    assert perclos([]) == 0.0


def test_all_closed():
    assert perclos([True, True, True]) == 1.0


def test_half_closed():
    assert perclos([True, False, True, False]) == 0.5
