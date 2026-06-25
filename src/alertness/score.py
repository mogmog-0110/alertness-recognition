"""予測結果CSVを、ルールと同じ指標で採点する。

ML/DL（Colab等）が出した予測をそのまま採点できる。CSVに「正解列」と「予測列」が
あればよく、特徴量の出どころ（自前/公開データセット）は問わない。採点の形式は
ルールベースと完全に同じなので、横並び比較ができる。

例:
    python -m alertness.score preds.csv
    python -m alertness.score preds.csv --true y_true --pred y_pred
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence

from .evaluation.metrics import format_scorecard, scorecard
from .evaluation.paths import resolve_csv_paths


def collect_labels(
    paths: Sequence[str], true_col: str, pred_col: str
) -> tuple[list[str], list[str]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                truth = (row.get(true_col) or "").strip()
                pred = (row.get(pred_col) or "").strip()
                if not truth or not pred:
                    continue
                y_true.append(truth)
                y_pred.append(pred)
    return y_true, y_pred


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="予測CSVをルールと同じ指標で採点する")
    parser.add_argument("csv", nargs="+", help="予測CSV（正解列と予測列を含む）")
    parser.add_argument("--true", default="label", help="正解ラベルの列名（既定: label）")
    parser.add_argument("--pred", default="pred", help="予測ラベルの列名（既定: pred）")
    parser.add_argument("--awake", default="awake", help="正常ラベル名（既定: awake）")
    args = parser.parse_args(argv)

    paths = resolve_csv_paths(args.csv, default_dir=".")
    y_true, y_pred = collect_labels(paths, args.true, args.pred)
    if not y_true:
        print(f"採点できる行がありません（'{args.true}' と '{args.pred}' 列を確認してください）。")
        return 1

    labels = sorted(set(y_true) | set(y_pred) | {args.awake})
    print(format_scorecard(scorecard(y_true, y_pred, labels, args.awake)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
