"""拍の並びから HRV（心拍変動）指標を計算する。numpy のみ。

入力は拍のピーク時刻[秒]の列。そこから RR 間隔（拍と拍の間隔）を作り、時間領域の
代表的な指標を出す。ストレスの段階化には主に RMSSD を使う（副交感神経の活動を反映し、
ストレスで低下しやすい）。値の単位はミリ秒で揃える（HRV の慣習）。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def rr_intervals_ms(peak_times_s: Sequence[float] | np.ndarray) -> np.ndarray:
    """拍のピーク時刻[秒]の列 → RR 間隔[ミリ秒]。拍が2つ未満なら空。"""
    times = np.asarray(peak_times_s, dtype=float).ravel()
    if times.size < 2:
        return np.empty(0, dtype=float)
    return np.diff(np.sort(times)) * 1000.0


def mean_hr(rr_ms: np.ndarray) -> float:
    """平均心拍[bpm]。RR が無ければ nan。"""
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size == 0 or np.mean(rr) <= 0:
        return float("nan")
    return float(60000.0 / np.mean(rr))


def sdnn(rr_ms: np.ndarray) -> float:
    """RR 間隔の標準偏差[ミリ秒]。全体的な変動の大きさ。"""
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size < 2:
        return float("nan")
    return float(np.std(rr, ddof=1))


def rmssd(rr_ms: np.ndarray) -> float:
    """隣り合う RR 差の二乗平均平方根[ミリ秒]。短期変動＝副交感神経の指標。"""
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size < 2:
        return float("nan")
    diff = np.diff(rr)
    return float(np.sqrt(np.mean(diff**2)))


def pnn50(rr_ms: np.ndarray) -> float:
    """隣接 RR 差が 50ms を超える割合(0..1)。RMSSD と同じく短期変動を見る補助指標。"""
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size < 2:
        return float("nan")
    diff = np.abs(np.diff(rr))
    return float(np.mean(diff > 50.0))
