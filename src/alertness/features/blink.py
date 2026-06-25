"""瞬きに関する時系列の計算。

入力は (時刻, 閉眼フラグ) の並び。しきい値判定はせず、数値だけを返す。
"""

from __future__ import annotations

from collections.abc import Sequence


def current_closed_duration(times: Sequence[float], closed: Sequence[bool]) -> float:
    """末尾から連続して閉じている時間（秒）。

    長く続くほどマイクロスリープ（一瞬の居眠り）の疑いが強い。
    """
    if not times or not closed or not closed[-1]:
        return 0.0
    start = times[-1]
    for t, c in zip(reversed(times), reversed(closed)):
        if not c:
            break
        start = t
    return max(0.0, times[-1] - start)


def blink_rate_per_minute(times: Sequence[float], closed: Sequence[bool]) -> float:
    """単位時間あたりの瞬き回数（毎分換算）。

    開→閉に変わった回数を立ち下がりとして数える。
    """
    if len(times) < 2:
        return 0.0
    elapsed = times[-1] - times[0]
    if elapsed <= 1e-6:
        return 0.0
    falling = sum(1 for i in range(1, len(closed)) if closed[i] and not closed[i - 1])
    return falling / elapsed * 60.0
