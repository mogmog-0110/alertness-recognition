"""CSVパス解決のテスト（cmdがワイルドカードを展開しない問題の対策）。"""

from __future__ import annotations

import os

from alertness.evaluation.paths import resolve_csv_paths


def test_directory_expands_to_csvs(tmp_path):
    (tmp_path / "a.csv").write_text("x", encoding="utf-8")
    (tmp_path / "b.csv").write_text("x", encoding="utf-8")
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")
    out = resolve_csv_paths([str(tmp_path)])
    assert len(out) == 2
    assert all(p.endswith(".csv") for p in out)


def test_explicit_file(tmp_path):
    f = tmp_path / "s.csv"
    f.write_text("x", encoding="utf-8")
    assert resolve_csv_paths([str(f)]) == [str(f)]


def test_glob_pattern(tmp_path):
    (tmp_path / "s1.csv").write_text("x", encoding="utf-8")
    (tmp_path / "s2.csv").write_text("x", encoding="utf-8")
    out = resolve_csv_paths([os.path.join(str(tmp_path), "s*.csv")])
    assert len(out) == 2


def test_deduplicates(tmp_path):
    f = tmp_path / "s.csv"
    f.write_text("x", encoding="utf-8")
    assert resolve_csv_paths([str(f), str(f)]) == [str(f)]
