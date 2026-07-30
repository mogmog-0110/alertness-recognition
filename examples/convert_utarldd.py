"""UTA-RLDD → manifest 変換器。動画1本まるごと1ラベルの公開データ向け。

UTA-RLDD (UTA Real-Life Drowsiness Dataset) は申請不要で公開されている眠気データセット。
60人 × 3本 = 180本、各約10分。ラベルは動画単位で alert(0) / low vigilant(5) / drowsy(10)
の3クラス。撮影は各自の携帯・webcam なので解像度とfpsはバラバラ（常に30fps未満）。

    https://sites.google.com/view/utarldd/home

## 配布物の構造についての前提

「被験者フォルダの中に 0.* / 5.* / 10.* という動画が1本ずつ」という構造を前提にする。
前提は LABEL_BY_STEM と walk() の2箇所に集めてあるので、配布物がこれと違えばそこだけ
書き換えればよい。ラベルを引けない動画があれば、黙って飛ばさずエラーで止める。

## アノテ規約: 3クラス → 4段階

正準ラベルは4段階なので、3クラスのどれかは必ず余る。ここでは low vigilant を medium に
写し、low を空ける。「low vigilant」は本人が運転を続けられる程度の低覚醒であり、警告の
第一段階(low)より一段重い状態と読む。逆に low へ寄せる規約もあり得るので、決めたら
docs/annotation-guide.md の眠気の表に書くこと。変更は LABEL_BY_STEM の一箇所で済む。

## このデータセットの弱点

10分の動画に1ラベルなので、フレーム単位ではラベルが相当ノイジーになる（drowsy 動画の
全フレームが drowsy 扱いになるが、実際には均一に眠いわけがない）。原論文のベースライン
実装も1分窓の瞬き特徴で扱っており、窓単位で使う前提のデータセットと考えたほうがよい。
フレーム単位でそのまま学習すると、眠気の精度が頭打ちになる原因になり得る。

## 使い方

    python examples\\convert_utarldd.py                    :: data/UTA-RLDD の全被験者
    python examples\\convert_utarldd.py data\\UTA-RLDD      :: 場所を指定
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from alertness.ingest.manifest import from_dict

# アノテ規約: 動画のファイル名(拡張子を除く) → 眠気の段階。
# UTA-RLDD のラベルはそのまま 0 / 5 / 10 という数値で表される。
LABEL_BY_STEM = {
    "0": "none",  # alert
    "5": "medium",  # low vigilant（low に寄せる規約もあり得る。上の説明を参照）
    "10": "high",  # drowsy
}
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")
# 用途タグ。UTA-RLDD は着座の自己撮影で運転ではないが、眠気は用途非依存の軸
# （colab の CONTEXT_FREE_AXES）なので空でよい。
CONTEXT = ""


def write_clip_manifest(path: Path, video: str, subject: str, context: str, **levels: str) -> Path:
    """動画1本＝1ラベルの manifest を書く。

    ingest.mapping.write_manifest は区間(segments)必須なので、動画単位ラベルには使えない。
    manifest.from_dict は segments が無く軸ラベルが直に載った形も受け付ける
    （manifest.py の「動画1本まるごと1ラベル（動画単位ラベルの公開データ向け）」経路）ので、
    その形を作って from_dict で検証してから書き出す。
    """
    data = {"video": video, "subject": subject, "context": context, **levels}
    from_dict(data)  # 書き出す前に妥当性を検証しておく
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def walk(root: Path) -> list[tuple[str, Path, str]]:
    """(被験者ID, 動画パス, 段階) の並びを返す。ここがデータセット固有の想定。

    被験者IDは動画の親フォルダ名。段階は動画のファイル名から引く。LABEL_BY_STEM に
    無い名前の動画は、ラベル不明として黙って飛ばさず呼び出し側に報告する。
    """
    found = []
    for video in sorted(root.rglob("*")):
        if video.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        stage = LABEL_BY_STEM.get(video.stem.strip())
        found.append((video.parent.name, video, stage or ""))
    return found


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "data/UTA-RLDD")
    if not root.is_dir():
        print(f"データセットのフォルダが見つかりません: {root}", file=sys.stderr)
        return 1

    found = walk(root)
    if not found:
        print(f"{root} に動画がありません（{VIDEO_SUFFIXES} を探しました）。", file=sys.stderr)
        return 1

    out_dir = Path("data/manifests")
    skipped = []
    for subject, video, stage in found:
        if not stage:
            skipped.append(video)
            continue
        path = write_clip_manifest(
            out_dir / f"utarldd_{subject}_{video.stem}.json",
            video=video.as_posix(),
            subject=f"rldd_{subject}",  # 他データセットと被験者IDが衝突しないよう接頭辞を付ける
            context=CONTEXT,
            drowsiness=stage,
        )
        print(f"{subject}/{video.name}: drowsiness={stage} → {path}")

    if skipped:
        print(
            f"\n⚠ ラベルを引けなかった動画が {len(skipped)} 本あります。"
            f"LABEL_BY_STEM の想定（{sorted(LABEL_BY_STEM)}）と実際のファイル名が違います:",
            file=sys.stderr,
        )
        for video in skipped[:5]:
            print(f"    {video}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
