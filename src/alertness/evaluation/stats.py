"""ラベルごとの特徴量分布を集計する。しきい値を「データから」決めるための材料。

例えば awake と drowsiness で ear_norm の中央値がどれだけ違うかが分かれば、
closed_ratio をどこに置けばよいかが読み取れる。外部依存なしの純粋処理。
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from collections.abc import Sequence

# 既定で見る列。判定に効く正規化特徴・軸スコア・各cueスコア。
# cue_* を見ると、どの手がかりがどのラベルで効く/誤発火するかが分かる。
DEFAULT_COLUMNS = (
    "ear_norm",
    "mar",
    "jawOpen",
    "pitch_rel",
    "yaw_rel",
    "gaze_off",
    "cue_eye_closure",
    "cue_blink",
    "cue_yawn",
    "cue_head_down",
    "cue_head_turn",
    "cue_gaze_off",
    "dim_drowsiness_score",
    "dim_distraction_score",
)


def _percentile(sorted_values: Sequence[float], ratio: float) -> float:
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, max(0, round(ratio * (len(sorted_values) - 1))))
    return sorted_values[index]


def summarize(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p10": _percentile(ordered, 0.1),
        "p90": _percentile(ordered, 0.9),
    }


def feature_stats(paths: Sequence[str], columns: Sequence[str] = DEFAULT_COLUMNS) -> dict:
    collected: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label = (row.get("label") or "").strip()
                if not label:
                    continue
                for column in columns:
                    raw = row.get(column, "")
                    if raw == "":
                        continue
                    try:
                        collected[label][column].append(float(raw))
                    except ValueError:
                        continue

    return {
        label: {column: summarize(values) for column, values in cols.items()}
        for label, cols in collected.items()
    }


def format_stats(stats: dict, columns: Sequence[str] = DEFAULT_COLUMNS) -> str:
    labels = sorted(stats)
    lines = []
    for column in columns:
        present = [label for label in labels if column in stats[label]]
        if not present:
            continue
        lines.append(f"== {column} ==")
        for label in present:
            s = stats[label][column]
            lines.append(
                f"  {label:14} n={s['n']:5d}  median {s['median']:.3f}"
                f"  p10 {s['p10']:.3f}  p90 {s['p90']:.3f}"
            )
    return "\n".join(lines)
