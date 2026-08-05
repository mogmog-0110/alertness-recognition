from __future__ import annotations

from collections.abc import Sequence


def classify_lod(scores: Sequence[float]) -> list[str]:
    """CDS を None / Low / Medium / High に変換する。"""
    levels: list[str] = []
    for score in scores:
        if score < 20:
            levels.append("none")
        elif score < 50:
            levels.append("low")
        elif score < 75:
            levels.append("medium")
        else:
            levels.append("high")
    return levels
