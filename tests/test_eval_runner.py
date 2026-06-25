"""評価ランナー（CSV→予測状態→採点）のテスト。"""

from __future__ import annotations

import csv

from alertness.evaluation.runner import (
    dimension_names_from_header,
    evaluate_files,
    predicted_state,
)


def test_dimension_names_from_header():
    fields = ["timestamp", "dim_drowsiness_score", "dim_drowsiness_level", "dim_distraction_level"]
    assert dimension_names_from_header(fields) == ["drowsiness", "distraction"]


def test_predicted_state_picks_highest_above_min():
    row = {"dim_drowsiness_level": "3", "dim_distraction_level": "1"}
    assert predicted_state(row, ["drowsiness", "distraction"], 2, "awake") == "drowsiness"


def test_predicted_state_awake_when_below_min():
    row = {"dim_drowsiness_level": "1", "dim_distraction_level": "0"}
    assert predicted_state(row, ["drowsiness", "distraction"], 2, "awake") == "awake"


def test_predicted_state_handles_missing_column():
    assert predicted_state({"dim_drowsiness_level": ""}, ["drowsiness"], 2, "awake") == "awake"


def test_evaluate_files(tmp_path):
    path = tmp_path / "session.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dim_drowsiness_level", "label"])
        writer.writeheader()
        writer.writerow({"dim_drowsiness_level": "3", "label": "drowsiness"})
        writer.writerow({"dim_drowsiness_level": "0", "label": "awake"})
    s = evaluate_files([str(path)], min_level=2, awake="awake")
    assert s["n"] == 2
    assert s["accuracy"] == 1.0
