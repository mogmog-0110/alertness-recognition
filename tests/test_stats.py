"""ラベル別の特徴量分布集計のテスト。"""

from __future__ import annotations

import csv

from alertness.evaluation.stats import feature_stats, summarize


def test_summarize_basic():
    s = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["n"] == 5
    assert s["median"] == 3.0


def test_summarize_empty():
    assert summarize([])["n"] == 0


def test_feature_stats_groups_by_label(tmp_path):
    path = tmp_path / "session.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ear_norm", "label"])
        writer.writeheader()
        writer.writerow({"ear_norm": "0.9", "label": "awake"})
        writer.writerow({"ear_norm": "1.1", "label": "awake"})
        writer.writerow({"ear_norm": "0.4", "label": "drowsiness"})

    stats = feature_stats([str(path)], ["ear_norm"])
    assert stats["awake"]["ear_norm"]["n"] == 2
    assert stats["awake"]["ear_norm"]["median"] == 1.0
    assert stats["drowsiness"]["ear_norm"]["n"] == 1
