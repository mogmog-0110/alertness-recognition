"""録画CSVの「分布」と「採点」をまとめて表示する。

analyze と eval はほぼセットで使うので1コマンドにした。
例:
    python -m alertness.report            # runs/ を自動で対象にする
    python -m alertness.report runs\\session.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .evaluation.metrics import format_scorecard, scorecard
from .evaluation.paths import resolve_csv_paths
from .evaluation.replay import replay_predict
from .evaluation.runner import evaluate_files
from .evaluation.stats import DEFAULT_COLUMNS, feature_stats, format_stats

_LEVELS = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _scorecard_text(paths: list[str], min_level: int, awake: str, config_path: str | None) -> str:
    if config_path is None:
        # 録画時に記録した判定をそのまま採点する。
        return format_scorecard(evaluate_files(paths, min_level, awake))
    # 現在の設定で再判定してから採点する（録り直さずに調整を試せる）。
    y_true, y_pred = replay_predict(paths, load_config(config_path), awake, min_level)
    labels = sorted(set(y_true) | set(y_pred) | {awake})
    return format_scorecard(scorecard(y_true, y_pred, labels, awake))


def _build_report(paths: list[str], min_level: int, awake: str, config_path: str | None) -> str:
    header = "=== 採点（再判定: 現設定）===" if config_path else "=== 採点（録画時の判定）==="
    return "\n".join(
        [
            f"対象: {len(paths)} ファイル",
            "",
            "=== ラベル別の特徴量分布 ===",
            format_stats(feature_stats(paths, DEFAULT_COLUMNS), DEFAULT_COLUMNS),
            "",
            header,
            _scorecard_text(paths, min_level, awake, config_path),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="録画CSVの分布と採点をまとめて表示する")
    parser.add_argument("csv", nargs="*", help="対象CSV/フォルダ（省略時は runs/）")
    parser.add_argument("--min-level", default="medium", choices=list(_LEVELS))
    parser.add_argument("--awake", default="awake")
    parser.add_argument("--out", default=None, help="結果をテキストファイルに保存する（UTF-8）")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="録画済み特徴量を現在の設定で再判定して採点する（録り直し不要）",
    )
    parser.add_argument("--config", default="config/default.yaml", help="--replay で使う設定")
    args = parser.parse_args(argv)

    paths = resolve_csv_paths(args.csv)
    if not paths:
        print("CSVが見つかりません。先に collect.bat で収録してください（runs/ に保存されます）。")
        return 1

    config_path = args.config if args.replay else None
    text = _build_report(paths, _LEVELS[args.min_level], args.awake, config_path)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n(保存しました: {args.out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
