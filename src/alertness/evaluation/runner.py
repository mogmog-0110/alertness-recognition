"""録画CSV（特徴量＋判定＋ラベル）を読み、ルールベースの予測を正解と突き合わせる。

予測状態は dim_*_level 列から決める（最も高いレベルの軸。どれも基準未満なら正常）。
正解は label 列。ML/DL は各モデルの予測列を作り、同じ metrics.py で採点すれば比較できる。
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence

from . import metrics


def dimension_names_from_header(fieldnames: Sequence[str] | None) -> list[str]:
    names = []
    for field in fieldnames or []:
        if field.startswith("dim_") and field.endswith("_level"):
            names.append(field[len("dim_") : -len("_level")])
    return names


def predicted_state(
    row: Mapping[str, str], dim_names: Sequence[str], min_level: int, awake: str
) -> str:
    best: str | None = None
    best_level = -1
    for name in dim_names:
        raw = row.get(f"dim_{name}_level", "")
        if raw == "":
            continue
        level = int(float(raw))
        if level > best_level:
            best_level = level
            best = name
    if best is None or best_level < min_level:
        return awake
    return best


def evaluate_files(paths: Sequence[str], min_level: int = 2, awake: str = "awake") -> dict:
    y_true: list[str] = []
    y_pred: list[str] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            dim_names = dimension_names_from_header(reader.fieldnames)
            for row in reader:
                label = (row.get("label") or "").strip()
                if not label:
                    continue  # ラベルなしの行は評価に使わない
                y_true.append(label)
                y_pred.append(predicted_state(row, dim_names, min_level, awake))

    labels = sorted(set(y_true) | set(y_pred) | {awake})
    return metrics.scorecard(y_true, y_pred, labels, negative_label=awake)
