from __future__ import annotations

import csv

from alertness.evaluation.segment import evaluate_axis_by_segment

_FIELDS = ["session_id", "label_drowsiness", "dim_drowsiness_level"]


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(level, dim_level):
    return {"session_id": "s1", "label_drowsiness": level, "dim_drowsiness_level": str(dim_level)}


def test_contiguous_levels_collapse_into_segments(tmp_path):
    rows = [_row("high", 3) for _ in range(3)]
    rows += [_row("none", 0) for _ in range(2)]
    path = tmp_path / "s1.csv"
    _write_csv(path, rows)

    score = evaluate_axis_by_segment([str(path)], "drowsiness")

    assert score["n"] == 2  # フレーム5行が2区間に畳まれる
    assert score["accuracy"] == 1.0
