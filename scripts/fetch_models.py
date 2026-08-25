"""顔ランドマークモデル(face_landmarker.task)を models/ に用意する。

scripts/setup.bat が curl でやっていることの、OS に依存しない版。scripts/ は他が
すべて .bat なので、mac / Linux にはモデルの取得経路が無かった。

    python scripts/fetch_models.py                  # 公式CDNから取得
    python scripts/fetch_models.py --from ~/dl.task # 手元のファイルから複製（オフライン用）
    python scripts/fetch_models.py --force          # 既にあっても取り直す

依存は標準ライブラリだけ。venv を作る前・依存を入れる前でも動く。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DEST = _ROOT / "models" / "face_landmarker.task"

# scripts/setup.bat が使っているものと同一の URL（MediaPipe 公式 float16/1、約3.7MB）。
_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker"
    "/face_landmarker/float16/1/face_landmarker.task"
)

# これを下回るファイルはモデルではなくエラーページ等とみなす。実物は約 3.7MB。
_MIN_BYTES = 1_000_000


def _validate(path: Path) -> None:
    size = path.stat().st_size
    if size < _MIN_BYTES:
        raise ValueError(
            f"取得したファイルが小さすぎます（{size} バイト）。中身を確認してください: {path}"
        )


def _install(source: Path | None, dest: Path) -> None:
    """一時ファイルに書いてから差し替える。

    途中で失敗したものが dest に残ると、次回以降 '既にある' と判定されて
    壊れたモデルを使い続けてしまうため、検証を通ってから初めて置く。
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        if source is None:
            print(f"[models] ダウンロード中: {_URL}")
            with urllib.request.urlopen(_URL, timeout=60) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out)
        else:
            print(f"[models] 複製中: {source}")
            shutil.copyfile(source, tmp)
        _validate(tmp)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="face_landmarker.task を models/ に用意する")
    parser.add_argument(
        "--from",
        dest="source",
        default=None,
        help="ダウンロードの代わりに複製する手元の .task ファイル",
    )
    parser.add_argument("--force", action="store_true", help="既にあっても取り直す")
    args = parser.parse_args(argv)

    if _DEST.exists() and not args.force:
        print(f"[models] 既にあります（--force で取り直し）: {_DEST}")
        return 0

    source = Path(args.source).expanduser() if args.source else None
    if source is not None and not source.is_file():
        print(f"[models] 複製元が見つかりません: {source}", file=sys.stderr)
        return 1

    _DEST.parent.mkdir(parents=True, exist_ok=True)
    try:
        _install(source, _DEST)
    except urllib.error.URLError as exc:
        print(f"[models] ダウンロードに失敗しました: {exc.reason}", file=sys.stderr)
        print(f"[models] 手元に .task があれば --from で渡せます。URL: {_URL}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"[models] 配置に失敗しました: {exc}", file=sys.stderr)
        return 1

    print(f"[models] 完了: {_DEST}（{_DEST.stat().st_size:,} バイト）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
