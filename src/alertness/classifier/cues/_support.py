"""cue 実装で共通して使う、時系列の取り出しヘルパ。"""

from __future__ import annotations

from collections.abc import Sequence

from ...contracts import Observation


def window_values(obs: Observation, key: str, seconds: float, default: float) -> tuple[
    list[float], list[float]
]:
    """直近 seconds 秒ぶんの (時刻リスト, 値リスト) を返す。顔なしフレームは除く。"""
    frames = [f for f in obs.history.recent(seconds) if f.face_present]
    times = [f.timestamp for f in frames]
    values = [f.get(key, default) for f in frames]
    return times, values


def trailing_true_seconds(times: Sequence[float], flags: Sequence[bool]) -> float:
    """末尾から連続して True が続いている時間（秒）。"""
    if not times or not flags or not flags[-1]:
        return 0.0
    start = times[-1]
    for t, c in zip(reversed(times), reversed(flags)):
        if not c:
            break
        start = t
    return max(0.0, times[-1] - start)


def true_fraction(flags: Sequence[bool]) -> float:
    """True の割合（0..1）。"""
    if not flags:
        return 0.0
    return sum(1 for c in flags if c) / len(flags)
