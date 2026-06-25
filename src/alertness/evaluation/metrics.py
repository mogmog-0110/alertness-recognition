"""判定の良し悪しを測る指標。

ここは外部依存なしの純粋関数にしてある。rule / ML / DL のどれでも、予測ラベル列
(y_pred) と正解ラベル列 (y_true) さえあれば同じコードで採点でき、比較が公平になる。
ラベル空間は「評価軸名 + 正常ラベル」（例: awake / drowsiness / distraction）。
"""

from __future__ import annotations

from collections.abc import Sequence


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        matrix[index[t]][index[p]] += 1
    return matrix


def _counts(y_true: Sequence[str], y_pred: Sequence[str], label: str) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == label and t == label:
            tp += 1
        elif p == label and t != label:
            fp += 1
        elif p != label and t == label:
            fn += 1
    return tp, fp, fn


def precision_recall_f1(
    y_true: Sequence[str], y_pred: Sequence[str], label: str
) -> tuple[float, float, float]:
    tp, fp, fn = _counts(y_true, y_pred, label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    return sum(precision_recall_f1(y_true, y_pred, label)[2] for label in labels) / len(labels)


def false_alarm_rate(
    y_true: Sequence[str], y_pred: Sequence[str], negative_label: str = "awake"
) -> float:
    # 正常なのに警告を出した割合（誤警告）。
    negatives = [(t, p) for t, p in zip(y_true, y_pred) if t == negative_label]
    if not negatives:
        return 0.0
    return sum(1 for _, p in negatives if p != negative_label) / len(negatives)


def miss_rate(
    y_true: Sequence[str], y_pred: Sequence[str], negative_label: str = "awake"
) -> float:
    # 異常を見逃して正常と判定した割合（見逃し）。
    positives = [(t, p) for t, p in zip(y_true, y_pred) if t != negative_label]
    if not positives:
        return 0.0
    return sum(1 for _, p in positives if p == negative_label) / len(positives)


def scorecard(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    negative_label: str = "awake",
) -> dict:
    per_class = {label: precision_recall_f1(y_true, y_pred, label) for label in labels}
    return {
        "n": len(y_true),
        "labels": list(labels),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "false_alarm_rate": false_alarm_rate(y_true, y_pred, negative_label),
        "miss_rate": miss_rate(y_true, y_pred, negative_label),
        "per_class": {
            label: {"precision": p, "recall": r, "f1": f} for label, (p, r, f) in per_class.items()
        },
        "confusion": confusion_matrix(y_true, y_pred, labels),
    }


def format_scorecard(s: dict) -> str:
    lines = [
        f"frames: {s['n']}",
        f"accuracy: {s['accuracy']:.3f}   macro-F1: {s['macro_f1']:.3f}",
        f"false-alarm: {s['false_alarm_rate']:.3f}   miss: {s['miss_rate']:.3f}",
        "per-class        precision recall    f1",
    ]
    for label in s["labels"]:
        c = s["per_class"][label]
        lines.append(f"  {label:14} {c['precision']:.3f}    {c['recall']:.3f}   {c['f1']:.3f}")
    lines.append("confusion (rows=true / cols=pred): " + ", ".join(s["labels"]))
    for label, row in zip(s["labels"], s["confusion"]):
        lines.append(f"  {label:14} " + " ".join(f"{v:5d}" for v in row))
    return "\n".join(lines)
