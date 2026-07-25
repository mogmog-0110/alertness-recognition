"""UBFC-Phys の zip を、置いてある分だけ端から自動で特徴量CSVにする。

ダウンロードは IEEE DataPort のログインが要るので手動（ブラウザやダウンロードマネージャで
sN.zip をフォルダに貯める）。この先――展開・被験者ふるい分け・取り込み・後片付け――を全自動で
回す。zip を1本ずつ処理し、既に取り込み済みの被験者は飛ばすので、追加ダウンロードのたびに
そのまま実行すればよい（何度実行しても安全）。

## ディスクを食わない順序

被験者ゲート（scenario と自己申告スコア）は info と selfReportedAnx だけで判定できる。
これらは1被験者100KB弱なので、まず小さいファイルだけ展開してゲートに掛け、不採用なら
14GB ある動画を展開しない。採用した被験者だけ動画を展開して取り込み、済んだら動画を消して
容量を戻す（--keep-videos で残せる）。取り込み済みの被験者を再実行したときも、残っている
展開動画があれば掃除する。

## zip の後片付け（容量対策）

CSV には生の幾何量と blendshape が全部入るので、一度取り込めば zip はほぼ不要。zip が
再び要るのは rPPG 設定や検出器を変えて動画から取り直すときだけ。毎回の最後に「もう消して
よい zip（取り込み済み or 不採用）」とその合計サイズを出す。--purge-zip を付けるとそれらを
実際に消す（小さいファイルは残すのでゲートの再判定はできる）。手元に全 56 本を置くと 800GB
超になるので、数本ずつダウンロード→処理→--purge-zip、を繰り返せばピーク容量を小さく保てる。

## 使い方

    python examples\\process_ubfc_phys.py                    :: data/UBFC-Phys の zip を順に処理
    python examples\\process_ubfc_phys.py --root D:\\ubfc      :: zip の置き場所を変える
    python examples\\process_ubfc_phys.py --keep-videos       :: 取り込み後も動画を残す
    python examples\\process_ubfc_phys.py --purge-zip         :: 処理し終えた zip を消す
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import convert_ubfc_phys as conv
import relabel_ubfc_eda as eda

from alertness.config import load_config
from alertness.ingest.manifest import load_manifest
from alertness.ingest.runner import run_ingest

SMALL_SUFFIXES = (".txt", ".csv")  # info / bvp / eda / selfReportedAnx。動画は .avi。


def _is_complete(zip_path: Path) -> bool:
    """ダウンロードが完了した健全な zip か。中身の目録が読めて動画が揃っていれば真。

    Chrome/Chrono は途中は別名(.crdownload)で書き、完了時に sN.zip へ改名するので、
    sN.zip が在る時点でほぼ完成しているが、改名直後や破損に備えて目録を確かめる。
    目録は末尾にあるので、書きかけの zip は ZipFile を開く時点で失敗する。
    """
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
    except (zipfile.BadZipFile, OSError):
        return False
    has_info = any(n.lower().endswith(".txt") for n in names)
    videos = [n for n in names if n.lower().endswith(".avi")]
    return has_info and len(videos) >= 3


def _extract(zip_path: Path, dst_root: Path, *, videos: bool) -> None:
    """zip の中身を dst_root/<subject>/ へ展開する。videos=False なら .avi を飛ばす。

    動画は数GBあるので、全体をメモリに読まずにストリームで書き出す。
    """
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
                shutil.copyfileobj(src, out)


def _already_ingested(out_base: Path, subject: str) -> bool:
    return any(out_base.glob(f"{subject}__*"))


def _videos_of(subject_dir: Path) -> list[Path]:
    return sorted(subject_dir.glob("*.avi"))


def process_subject(
    zip_path: Path, config: dict, root: Path, out_base: Path, *, keep_videos: bool
) -> tuple[str, str]:
    """1本の zip を処理し、(状態, メッセージ) を返す。

    状態は "ingested" / "rejected" / "skipped" / "pending"。前3つは zip をもう消してよい。
    "pending"（ダウンロード中）と "failed"（処理中の例外）は消さずに残す。
    """
    subject = zip_path.stem  # s9.zip → s9
    subject_dir = root / subject
    if _already_ingested(out_base, subject):
        # 済んでいるので飛ばすが、前回の展開動画が残っていれば掃除する。
        removed = [v for v in _videos_of(subject_dir)]
        for video in removed:
            video.unlink()
        note = f"（残っていた動画 {len(removed)} 本を掃除）" if removed else ""
        return "skipped", f"{subject}: 取り込み済み。飛ばす{note}"

    if not _is_complete(zip_path):
        return "pending", f"{subject}: ダウンロード中/不完全。次回に回す"

    subject_dir.mkdir(parents=True, exist_ok=True)
    _extract(zip_path, subject_dir, videos=False)  # まず小さいファイルだけ

    info = conv.read_info(subject_dir / f"info_{subject}.txt")
    anxiety = conv.read_anxiety(subject_dir / f"selfReportedAnx_{subject}.csv")
    reason = conv.induction_worked(info, anxiety)
    if reason:
        return "rejected", f"{subject}: 不採用（{reason}）。動画は展開しない"

    manifests = conv.convert_subject(root, subject, Path("data/manifests"), force=False)
    _extract(zip_path, subject_dir, videos=True)  # 採用したので動画を展開

    for manifest_path in manifests:
        run_ingest(config, load_manifest(manifest_path), out_base)
    if not keep_videos:
        for video in _videos_of(subject_dir):
            video.unlink()

    # 3タスク揃ったので、EDA(eda_*.csv)から窓単位ラベルを足す（動画は不要）。非反応者は
    # ラベルの根拠が無いので EDA ラベルだけ付けない（プロトコルラベルは残る）。
    eda_count, eda_note = eda.relabel_subject(subject, out_base, root)
    eda_msg = f"EDAラベル {eda_count}本" if eda_count else f"EDAラベル無し（{eda_note}）"

    tail = "" if keep_videos else "（動画は削除）"
    return "ingested", f"{subject}: 取り込み {len(manifests)} タスク / {eda_msg}{tail}"


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
    parser.add_argument("--purge-zip", action="store_true", help="処理し終えた zip を削除する")
    args = parser.parse_args(argv)

    root = Path(args.root)
    zips = sorted(root.glob("s*.zip"), key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 0)
    if not zips:
        print(f"{root} に sN.zip がありません。ダウンロードしましたか？", file=sys.stderr)
        return 1

    config = load_config(args.config)
    out_base = Path(args.out)
    done: list[Path] = []  # もう消してよい zip（取り込み済み or 不採用）
    for zip_path in zips:
        try:
            status, message = process_subject(
                zip_path, config, root, out_base, keep_videos=args.keep_videos
            )
            print(message)
            if status in ("ingested", "rejected", "skipped"):
                done.append(zip_path)
        except Exception as exc:  # 1本の失敗で全体を止めない
            print(f"{zip_path.stem}: 失敗 - {exc}", file=sys.stderr)

    print(_summary(out_base))
    _report_disk(done, purge=args.purge_zip)
    return 0


def _report_disk(done: list[Path], *, purge: bool) -> None:
    """処理し終えて消してよい zip を報告する。purge ならその場で消す。"""
    if not done:
        return
    total_gb = sum(p.stat().st_size for p in done) / 1e9
    if purge:
        for zip_path in done:
            zip_path.unlink()
        print(f"処理済みの zip {len(done)} 本を削除しました（{total_gb:.0f}GB を確保）。")
    else:
        names = ", ".join(p.name for p in done)
        print(
            f"もう消してよい zip: {len(done)} 本 / 計 {total_gb:.0f}GB → {names}\n"
            "  （--purge-zip で自動削除。CSV と小さいファイルは残るので再判定はできる）"
        )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
