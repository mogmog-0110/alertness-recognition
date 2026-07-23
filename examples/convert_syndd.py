"""SynDD1/SynDD2 → manifest 変換器。gaze zone の区間アノテを注意逸脱の段階へ写す。

SynDD は Mendeley Data で CC BY 公開されている運転者モニタリングのデータセット（申請不要）。
MP4 / 1920x1080 / 30fps、車内3カメラ（ダッシュボード / ルームミラー付近 / 右上窓角）。
18の逸脱行動と11の gaze zone が、start / end 時刻（h:mm:ss）付きのCSVで配布される。

    https://data.mendeley.com/datasets/ptcp7rp3wb/4

区間アノテなので、repo の manifest の segments 形式にほぼ1:1で写せる。gaze zone は
gaze_off / head_turn cue が見ているものと概念的に同じなので、ルール判定との突き合わせにも使える。

## ⚠ 未検証の想定（実データ入手後にここだけ直す）

アノテCSVの列名と gaze zone の表記は手元で未確認。想定を3箇所に集めてある:
  - COLUMN_CANDIDATES … 列名の候補（大小文字・空白・アンダースコアは無視して照合）
  - GAZE_ZONE_TO_STAGE … zone名 → 段階。これがアノテ規約そのもの
  - VIEW … 3カメラのうちどれを使うか
表記が違えば「未知のzone」として**エラーで止まる**。黙って none に落とすとラベルが静かに
汚染され、後から気づけないため。

## アノテ規約: gaze zone → 注意逸脱の段階

「前方から目を離している時間が長いほど、かつ運転と無関係な対象ほど重い」という向きで写す。
運転に必要な確認（ミラー・メーター）は前方を見ていなくても軽い扱いにする。この考え方は
attention_buffer cue の根拠（前方から2秒視線を外すと車線内の位置把握が崩れる）と揃えてある。

段階の意味は docs/annotation-guide.md の「注意逸脱 / context: driving」の表に書くこと。

## 使い方

    python examples\\convert_syndd.py data\\SynDD1
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from alertness.ingest.mapping import segment, write_manifest

# アノテ規約: gaze zone → 注意逸脱の段階。キーは小文字・記号を潰した形で照合する。
GAZE_ZONE_TO_STAGE = {
    # 前方。逸脱なし。
    "forward": "none",
    "forwardwindow": "none",
    "windshield": "none",
    "road": "none",
    # 運転に必要な確認。前方からは外れているが目的がある。
    "speedometer": "low",
    "instrumentcluster": "low",
    "rearviewmirror": "low",
    "leftmirror": "low",
    "rightmirror": "low",
    "sidemirror": "low",
    # 運転と無関係だが車内前方。
    "radio": "medium",
    "centerconsole": "medium",
    "centerstack": "medium",
    "passenger": "medium",
    "leftwindow": "medium",
    "rightwindow": "medium",
    # 前方から大きく外れる。もっとも重い。
    "phone": "high",
    "cellphone": "high",
    "lap": "high",
    "down": "high",
    "backseat": "high",
}

# アノテCSVの列名の候補。実物に合わせて足す。
COLUMN_CANDIDATES = {
    "start": ("starttime", "start", "begin", "begintime"),
    "end": ("endtime", "end", "stop", "stoptime"),
    "zone": ("gazezone", "zone", "gaze", "label", "activity"),
}

# 3カメラのうち、顔が正面寄りに写る視点を使う。実物を見て決める。
VIEW = "Dashboard"
CONTEXT = "driving"  # 注意逸脱は用途依存の軸なので、用途タグを必ず入れる


def _key(name: str) -> str:
    # 大小文字・空白・アンダースコア・ハイフンを潰して照合する。
    return "".join(ch for ch in name.lower() if ch.isalnum())


def parse_hms(value: str) -> float:
    """h:mm:ss（または mm:ss / 秒）を秒に直す。"""
    parts = [p.strip() for p in str(value).strip().split(":")]
    try:
        numbers = [float(p) for p in parts if p != ""]
    except ValueError as exc:
        raise ValueError(f"時刻として読めません: {value!r}") from exc
    if not numbers or len(numbers) > 3:
        raise ValueError(f"時刻として読めません: {value!r}")
    seconds = 0.0
    for n in numbers:  # 先頭から順に 60 倍していけば h:mm:ss でも mm:ss でも秒でも通る
        seconds = seconds * 60.0 + n
    return seconds


def resolve_columns(fieldnames: list[str]) -> dict[str, str]:
    """CSVの実際の列名を、こちらの役割名（start/end/zone）に対応づける。"""
    by_key = {_key(name): name for name in fieldnames}
    resolved = {}
    for role, candidates in COLUMN_CANDIDATES.items():
        hit = next((by_key[c] for c in candidates if c in by_key), None)
        if hit is None:
            raise ValueError(
                f"'{role}' に当たる列がCSVにありません。実際の列: {fieldnames}\n"
                f"  COLUMN_CANDIDATES['{role}'] に実物の列名を足してください。"
            )
        resolved[role] = hit
    return resolved


def read_segments(csv_path: Path) -> list[dict]:
    """アノテCSV → 区間ラベルの並び。未知の zone があれば止める。"""
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSVにヘッダがありません: {csv_path}")
        cols = resolve_columns(list(reader.fieldnames))
        rows = list(reader)

    segments, unknown = [], set()
    for row in rows:
        zone = str(row[cols["zone"]]).strip()
        stage = GAZE_ZONE_TO_STAGE.get(_key(zone))
        if stage is None:
            unknown.add(zone)
            continue
        start, end = parse_hms(row[cols["start"]]), parse_hms(row[cols["end"]])
        if start >= end:
            continue  # 長さ0の区間は manifest 側で弾かれるので、ここで落とす
        segments.append(segment(start, end, distraction=stage))

    if unknown:
        raise ValueError(
            f"GAZE_ZONE_TO_STAGE に無い zone があります: {sorted(unknown)}\n"
            f"  黙って none に落とすとラベルが静かに汚染されるので止めました。"
            f"  規約を決めて GAZE_ZONE_TO_STAGE に足してください（{csv_path}）。"
        )
    return segments


def find_video(csv_path: Path, view: str) -> Path | None:
    """アノテCSVと同じフォルダから、指定の視点の動画を探す。"""
    candidates = [p for p in sorted(csv_path.parent.glob("*.mp4")) if _key(view) in _key(p.name)]
    return candidates[0] if candidates else None


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "data/SynDD1")
    if not root.is_dir():
        print(f"データセットのフォルダが見つかりません: {root}", file=sys.stderr)
        return 1

    csv_files = sorted(root.rglob("*.csv"))
    if not csv_files:
        print(f"{root} にアノテCSVがありません。", file=sys.stderr)
        return 1

    out_dir = Path("data/manifests")
    for csv_path in csv_files:
        video = find_video(csv_path, VIEW)
        if video is None:
            print(f"⚠ {csv_path.parent} に視点 '{VIEW}' の mp4 が見つかりません。飛ばします。")
            continue
        segments = read_segments(csv_path)
        if not segments:
            print(f"⚠ {csv_path}: 区間を1つも作れませんでした。飛ばします。")
            continue
        subject = f"syndd_{csv_path.parent.name}"
        path = write_manifest(
            out_dir / f"syndd_{csv_path.parent.name}_{video.stem}.json",
            video=video.as_posix(),
            subject=subject,
            context=CONTEXT,
            segments=segments,
        )
        print(f"{subject}: {len(segments)}区間 → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
