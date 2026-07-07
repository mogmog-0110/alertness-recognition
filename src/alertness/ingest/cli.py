"""取り込みのコマンドライン。

例:
    python -m alertness.ingest --manifests data/manifests --out runs/ingested
    python -m alertness.ingest --manifests clip.json
--manifests には JSON 単体でもフォルダでも渡せる。どのデータセットかは問わない
（manifest の形にさえしてあればよい）。各動画を特徴量CSVに変換して書き出す。
"""

from __future__ import annotations

import argparse

from ..config import load_config
from .manifest import manifests_from
from .runner import run_ingest_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="外部データセットを特徴量CSVに取り込む")
    parser.add_argument("--config", default="config/default.yaml", help="設定ファイル")
    parser.add_argument(
        "--manifests", required=True, help="manifest のJSON、またはそれを集めたフォルダ"
    )
    parser.add_argument("--out", default="runs/ingested", help="CSV出力先の基底ディレクトリ")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    directories = run_ingest_all(config, manifests_from(args.manifests), args.out)

    if not directories:
        print("取り込む動画がありませんでした。")
        return 1
    for directory in directories:
        print(f"取り込み完了 -> {directory}")
    print(f"合計 {len(directories)} 本を取り込みました。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
