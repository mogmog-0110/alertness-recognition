"""録画CSVをルールベース予測で採点する。

例:
    python -m alertness.evaluate runs\\session_1.csv runs\\session_2.csv
複数ファイルはまとめて1つの成績として集計する。ML/DL も同じ metrics で採点し比較する。
"""

from __future__ import annotations

import argparse

from .evaluation.metrics import format_scorecard
from .evaluation.paths import resolve_csv_paths
from .evaluation.runner import evaluate_files

_LEVELS = {"none": 0, "low": 1, "medium": 2, "high": 3}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="録画CSVをルールベース予測で採点する")
    parser.add_argument("csv", nargs="*", help="評価するCSV/フォルダ（省略時は runs/）")
    parser.add_argument(
        "--min-level",
        default="medium",
        choices=list(_LEVELS),
        help="警告とみなす最小レベル（既定: medium）",
    )
    parser.add_argument("--awake", default="awake", help="正常状態のラベル名（既定: awake）")
    args = parser.parse_args(argv)

    paths = resolve_csv_paths(args.csv)
    if not paths:
        print("CSVが見つかりません。collect.bat で収録してください（runs/ に保存されます）。")
        return 1
    print(format_scorecard(evaluate_files(paths, _LEVELS[args.min_level], args.awake)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
