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


def plausible_rr(
    rr_ms: np.ndarray,
    min_ms: float = 300.0,
    max_ms: float = 2000.0,
    max_deviation: float = 0.25,
) -> np.ndarray:
    """各 RR 間隔がもっともらしいかの真偽列。

    拍を1つ落とすと、その間隔だけが 2 倍になる。RMSSD は隣接 RR の差の二乗平均なので、
    1回の取りこぼしが値を跳ね上げる。指標を出す前にこうした異常値を外すのは HRV 解析の
    標準的な作法で、これを省くと「変動が大きい＝リラックス」と正反対に読める値が出る。

    判定は2段階。生理的な範囲（min_ms〜max_ms）から外れたものを先に落とし、残りの
    中央値から max_deviation を超えて離れたものを落とす。中央値を使うのは、異常値が
    混じったままの平均を基準にすると、その異常値自身が基準を引き寄せてしまうため。
    """
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size == 0:
        return np.empty(0, dtype=bool)
    in_range = (rr >= min_ms) & (rr <= max_ms)
    if not np.any(in_range):
        return in_range
    center = float(np.median(rr[in_range]))
    if center <= 0:
        return in_range
    return in_range & (np.abs(rr - center) <= max_deviation * center)


def rmssd(rr_ms: np.ndarray, valid: np.ndarray | None = None) -> float:
    """隣り合う RR 差の二乗平均平方根[ミリ秒]。短期変動＝副交感神経の指標。

    valid を渡すと、両隣とも妥当な組だけで計算する。異常値を列から取り除いて詰めると、
    時間的に隣り合っていない拍どうしの差を「隣接差」として数えてしまう。
    使える組が無ければ nan（0 ではない。測れないことと変動が無いことは別）。
    """
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size < 2:
        return float("nan")
    diff = np.diff(rr)
    if valid is not None:
        pairs = np.asarray(valid, dtype=bool)
        usable = pairs[:-1] & pairs[1:]
        if not np.any(usable):
            return float("nan")
        diff = diff[usable]
    return float(np.sqrt(np.mean(diff**2)))


def pnn50(rr_ms: np.ndarray) -> float:
    """隣接 RR 差が 50ms を超える割合(0..1)。RMSSD と同じく短期変動を見る補助指標。"""
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size < 2:
        return float("nan")
    diff = np.abs(np.diff(rr))
    return float(np.mean(diff > 50.0))
