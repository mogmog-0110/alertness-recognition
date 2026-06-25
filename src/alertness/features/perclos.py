"""PERCLOS（一定時間に占める閉眼の割合）の計算。

時系列の計算だが、しきい値判定はせず割合だけを返す純粋関数にしてある。
閉眼かどうかの判定は呼び出し側（cue）が基準値を使って行う。
"""

from __future__ import annotations

from collections.abc import Sequence


def perclos(closed_flags: Sequence[bool]) -> float:
    """閉眼フラグ列のうち、閉眼が占める割合（0..1）。

    高いほど眠気の疑いが強い。空列なら 0 を返す。
    """
    if not closed_flags:
        return 0.0
    closed = sum(1 for c in closed_flags if c)
    return closed / len(closed_flags)
