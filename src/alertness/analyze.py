"""録画CSVからラベルごとの特徴量分布を出す。しきい値調整の手がかりに使う。

例:
    python -m alertness.analyze runs\\session_*.csv
"""

from __future__ import annotations

import argparse

from .evaluation.paths import resolve_csv_paths
from .evaluation.stats import DEFAULT_COLUMNS, feature_stats, format_stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ラベル別の特徴量分布を表示する")
    parser.add_argument("csv", nargs="*", help="分析するCSV/フォルダ（省略時は runs/）")
    parser.add_argument(
        "--columns",
        nargs="*",
        default=list(DEFAULT_COLUMNS),
        help="見たい列（省略時は主要な特徴量）",
    )
    args = parser.parse_args(argv)

    paths = resolve_csv_paths(args.csv)
    stats = feature_stats(paths, args.columns)
    if not stats:
        print("ラベル付きの行が見つかりませんでした（録画中に数字キーでラベルを付けてください）。")
        return 1
    print(format_stats(stats, args.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
