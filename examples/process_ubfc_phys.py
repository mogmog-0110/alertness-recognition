"""UBFC-Phys の zip を、置いてある分だけ端から自動で特徴量CSVにする。

ダウンロードは IEEE DataPort のログインが要るので手動（ブラウザやダウンロードマネージャで
sN.zip をフォルダに貯める）。この先――展開・被験者ふるい分け・取り込み・後片付け――を全自動で
回す。zip を1本ずつ処理し、既に取り込み済みの被験者は飛ばすので、追加ダウンロードのたびに
そのまま実行すればよい（何度実行しても安全）。

## ディスクを食わない順序

被験者ゲート（scenario と自己申告スコア）は info と selfReportedAnx だけで判定できる。
これらは1被験者100KB弱なので、まず小さいファイルだけ展開してゲートに掛け、不採用なら
14GB ある動画を展開しない。採用した被験者だけ動画を展開して取り込み、済んだら動画を消して
容量を戻す（--keep-videos で残せる）。zip 自体は消さない（再取り込みの元として手元に残す）。

## 使い方

    python examples\\process_ubfc_phys.py                    :: data/UBFC-Phys の zip を順に処理
    python examples\\process_ubfc_phys.py --root D:\\ubfc      :: zip の置き場所を変える
    python examples\\process_ubfc_phys.py --keep-videos       :: 取り込み後も動画を残す
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import convert_ubfc_phys as conv

from alertness.config import load_config
from alertness.ingest.manifest import load_manifest
from alertness.ingest.runner import run_ingest

SMALL_SUFFIXES = (".txt", ".csv")  # info / bvp / eda / selfReportedAnx。動画は .avi。


def _extract(zip_path: Path, dst_root: Path, *, videos: bool) -> None:
    """zip の中身を dst_root/<subject>/ へ展開する。videos=False なら .avi を飛ばす。"""
    with zipfile.ZipFile(zip_path) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            is_video = entry.filename.lower().endswith(".avi")
            if is_video and not videos:
                continue
            if not is_video and not entry.filename.lower().endswith(SMALL_SUFFIXES):
                continue
            target = dst_root / Path(entry.filename).name
            if target.exists() and target.stat().st_size == entry.file_size:
                continue  # 展開済みは飛ばす
            with archive.open(entry) as src, target.open("wb") as out:
                out.write(src.read())


def _already_ingested(out_base: Path, subject: str) -> bool:
    return any(out_base.glob(f"{subject}__*"))


def _videos_of(subject_dir: Path) -> list[Path]:
    return sorted(subject_dir.glob("*.avi"))


def process_subject(
    zip_path: Path, config: dict, root: Path, out_base: Path, *, keep_videos: bool
) -> str:
    """1本の zip を処理し、結果を一言で返す（skipped / rejected / ingested）。"""
    subject = zip_path.stem  # s9.zip → s9
    if _already_ingested(out_base, subject):
        return f"{subject}: 取り込み済み。飛ばす"

    subject_dir = root / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    _extract(zip_path, subject_dir, videos=False)  # まず小さいファイルだけ

    info = conv.read_info(subject_dir / f"info_{subject}.txt")
    anxiety = conv.read_anxiety(subject_dir / f"selfReportedAnx_{subject}.csv")
    reason = conv.induction_worked(info, anxiety)
    if reason:
        return f"{subject}: 不採用（{reason}）。動画は展開しない"

    manifests = conv.convert_subject(root, subject, Path("data/manifests"), force=False)
    _extract(zip_path, subject_dir, videos=True)  # 採用したので動画を展開

    for manifest_path in manifests:
        run_ingest(config, load_manifest(manifest_path), out_base)
    if not keep_videos:
        for video in _videos_of(subject_dir):
            video.unlink()
    tail = "" if keep_videos else "（動画は削除）"
    return f"{subject}: 取り込み {len(manifests)} タスク{tail}"


def _summary(out_base: Path) -> str:
    import csv
    from collections import Counter

    subjects, stages = set(), Counter()
    for path in out_base.glob("*/*.csv"):
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            subjects.add(rows[0]["subject"])
            stages[rows[0].get("label_stress", "")] += 1
    return f"取り込み済み被験者 {len(subjects)}人 / クリップのラベル内訳 {dict(stages)}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="UBFC-Phys の zip を自動で特徴量CSVにする")
    parser.add_argument("--root", default="data/UBFC-Phys", help="sN.zip の置き場所")
    parser.add_argument("--config", default="config/default.yaml", help="設定ファイル")
    parser.add_argument("--out", default="runs/ingested", help="CSV出力先")
    parser.add_argument("--keep-videos", action="store_true", help="取り込み後も動画を残す")
    args = parser.parse_args(argv)

    root = Path(args.root)
    zips = sorted(root.glob("s*.zip"), key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 0)
    if not zips:
        print(f"{root} に sN.zip がありません。ダウンロードしましたか？", file=sys.stderr)
        return 1

    config = load_config(args.config)
    out_base = Path(args.out)
    for zip_path in zips:
        try:
            print(process_subject(zip_path, config, root, out_base, keep_videos=args.keep_videos))
        except Exception as exc:  # 1本の失敗で全体を止めない
            print(f"{zip_path.stem}: 失敗 - {exc}", file=sys.stderr)
    print(_summary(out_base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
