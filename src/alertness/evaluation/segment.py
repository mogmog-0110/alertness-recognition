"""区間単位・軸別の採点。フレーム単位の予測を区間ごとに集約してから軸別に突き合わせる。

眠気などは秒〜分でゆっくり変わる状態なので、フレーム単位だと細かく採点しすぎる。
連続する同一ラベルを1区間にまとめ、その区間の予測は多数決で1つに落とす。
軸（drowsiness / distraction）ごとに、正解の段階と `dim_<軸>_level` を突き合わせる。
段階を4クラスとして metrics.py をそのまま使うので、混同行列で隣接誤りも見える。
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence

from . import metrics

_LEVEL_NAMES = ("none", "low", "medium", "high")


def _predicted_level(row: Mapping[str, str], axis: str) -> str:
    raw = row.get(f"dim_{axis}_level", "")
    if raw == "":
        return "none"
    return _LEVEL_NAMES[int(float(raw))]


def _labeled_segments(
    rows: Sequence[Mapping[str, str]], axis: str
) -> Iterator[tuple[str, list[Mapping[str, str]]]]:
    key_col = f"label_{axis}"
    current_key: tuple[str, str] | None = None
    bucket: list[Mapping[str, str]] = []
    for row in rows:
        level = (row.get(key_col) or "").strip()
        if not level:
            if bucket:
                yield bucket[0][key_col].strip(), bucket
                bucket = []
            current_key = None
            continue
        key = (row.get("session_id", ""), level)
        if key != current_key and bucket:
            yield bucket[0][key_col].strip(), bucket
            bucket = []
        current_key = key
        bucket.append(row)
    if bucket:
        yield bucket[0][key_col].strip(), bucket


def evaluate_axis_by_segment(paths: Sequence[str], axis: str, awake: str = "none") -> dict:
    y_true: list[str] = []
    y_pred: list[str] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for true_level, seg_rows in _labeled_segments(rows, axis):
            preds = [_predicted_level(r, axis) for r in seg_rows]
            y_true.append(true_level)
            y_pred.append(Counter(preds).most_common(1)[0][0])

    labels = sorted(set(y_true) | set(y_pred) | {awake})
    return metrics.scorecard(y_true, y_pred, labels, negative_label=awake)
