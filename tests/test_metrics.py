"""評価指標のテスト。"""

from __future__ import annotations

from alertness.evaluation import metrics


def test_perfect_prediction():
    y = ["awake", "drowsiness", "distraction"]
    s = metrics.scorecard(y, list(y), ["awake", "distraction", "drowsiness"])
    assert s["accuracy"] == 1.0
    assert s["macro_f1"] == 1.0
    assert s["false_alarm_rate"] == 0.0
    assert s["miss_rate"] == 0.0


def test_false_alarm_and_miss():
    y_true = ["awake", "awake", "drowsiness", "drowsiness"]
    y_pred = ["awake", "drowsiness", "awake", "drowsiness"]
    s = metrics.scorecard(y_true, y_pred, ["awake", "drowsiness"])
    assert s["false_alarm_rate"] == 0.5  # 正常2件のうち1件を誤警告
    assert s["miss_rate"] == 0.5  # 異常2件のうち1件を見逃し


def test_precision_recall_f1():
    y_true = ["a", "a", "b"]
    y_pred = ["a", "b", "b"]
    p, r, f = metrics.precision_recall_f1(y_true, y_pred, "a")
    assert p == 1.0
    assert r == 0.5
    assert abs(f - 2 / 3) < 1e-9


def test_confusion_matrix():
    matrix = metrics.confusion_matrix(["a", "b"], ["a", "a"], ["a", "b"])
    assert matrix == [[1, 0], [1, 0]]


def test_format_scorecard_runs():
    s = metrics.scorecard(["awake"], ["awake"], ["awake"])
    assert "accuracy" in metrics.format_scorecard(s)


def test_ordinal_mae_counts_stage_distance():
    # high を medium と間違えれば1段、none と間違えれば3段。平均で 2 段ずれる。
    mae = metrics.ordinal_mae(["high", "high"], ["medium", "none"])
    assert mae == 2.0


def test_adjacent_accuracy_allows_one_stage_off():
    # medium→low(1段) は許容、medium→none(2段) は不可 → 1/2。
    acc = metrics.adjacent_accuracy(["medium", "medium"], ["low", "none"])
    assert acc == 0.5


def test_ordinal_metrics_none_for_non_stage_labels():
    # 段階でないラベル（awake/drowsiness）では順序を測れないので None。
    assert metrics.ordinal_mae(["awake"], ["drowsiness"]) is None
    assert metrics.adjacent_accuracy(["awake"], ["drowsiness"]) is None


def test_scorecard_includes_ordinal_for_stage_labels():
    stages = ["none", "low", "medium", "high"]
    s = metrics.scorecard(["none", "high"], ["low", "medium"], stages, negative_label="none")
    assert s["ordinal_mae"] == 1.0  # none→low=1, high→medium=1
    assert s["adjacent_accuracy"] == 1.0
