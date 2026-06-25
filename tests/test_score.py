"""予測CSV採点（ML/DLの結果を同じ指標で採点）のテスト。"""

from __future__ import annotations

import csv

from alertness.score import collect_labels, main


def _write(path, rows, fields=("label", "pred")):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def test_collect_labels_skips_blank(tmp_path):
    path = tmp_path / "p.csv"
    _write(
        path,
        [
            {"label": "awake", "pred": "awake"},
            {"label": "drowsiness", "pred": "awake"},
            {"label": "", "pred": "awake"},  # 空はスキップ
        ],
    )
    y_true, y_pred = collect_labels([str(path)], "label", "pred")
    assert y_true == ["awake", "drowsiness"]
    assert y_pred == ["awake", "awake"]


def test_main_prints_scorecard(tmp_path, capsys):
    path = tmp_path / "p.csv"
    _write(path, [{"label": "awake", "pred": "awake"}, {"label": "drowsiness", "pred": "drowsiness"}])
    rc = main([str(path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "accuracy: 1.000" in out


def test_main_custom_columns(tmp_path, capsys):
    path = tmp_path / "p.csv"
    _write(
        path,
        [{"y_true": "awake", "y_pred": "drowsiness"}],
        fields=("y_true", "y_pred"),
    )
    rc = main([str(path), "--true", "y_true", "--pred", "y_pred"])
    assert rc == 0
    assert "false-alarm: 1.000" in capsys.readouterr().out
