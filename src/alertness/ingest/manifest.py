"""取り込み用の正規化スキーマ。動画1本と、その時間区間ごとの軸別・段階ラベル。

正準ラベルは2軸（drowsiness / distraction）× 4段階（none/low/medium/high）。
外部データセットは配布形式がバラバラなので、まずこの共通形に落としてから特徴抽出に渡す。
配布形式→この形への変換は核の外で行い、核には manifest だけが渡る。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

LEVELS = ("none", "low", "medium", "high")
AXES = ("drowsiness", "distraction")


@dataclass(frozen=True)
class Segment:
    start: float  # 秒
    end: float  # 秒（この時刻は含まない）
    drowsiness: str  # LEVELS のいずれか
    distraction: str


@dataclass(frozen=True)
class ClipManifest:
    video: str
    subject: str
    context: str  # 用途（driving / study 等）。空でもよい
    segments: tuple[Segment, ...]

    def labels_at(self, timestamp: float) -> dict[str, str]:
        # 区間に当たらない時刻は空＝無ラベル（採点対象外）として扱う。
        for seg in self.segments:
            if seg.start <= timestamp < seg.end:
                return {"drowsiness": seg.drowsiness, "distraction": seg.distraction}
        return {"drowsiness": "", "distraction": ""}


def _level(value: object) -> str:
    v = str(value).lower()
    if v not in LEVELS:
        raise ValueError(f"レベルは {LEVELS} のいずれかである必要があります: {value}")
    return v


def _axes(data: dict) -> tuple[str, str]:
    # 指定の無い軸は none（例: 眠気データに注意逸脱の情報が無い場合）。
    return _level(data.get("drowsiness", "none")), _level(data.get("distraction", "none"))


def from_dict(data: dict) -> ClipManifest:
    video = data.get("video")
    if not video:
        raise ValueError("manifest に video がありません。")
    subject = str(data.get("subject", "default"))
    context = str(data.get("context", ""))

    raw_segments = data.get("segments")
    if raw_segments:
        segments = []
        for s in raw_segments:
            start, end = float(s["start"]), float(s["end"])
            if start >= end:
                raise ValueError(f"区間の start は end より前である必要があります: {s}")
            drowsiness, distraction = _axes(s)
            segments.append(Segment(start, end, drowsiness, distraction))
        segments = tuple(segments)
    elif "drowsiness" in data or "distraction" in data:
        # 動画1本まるごと1ラベル（動画単位ラベルの公開データ向け）。
        drowsiness, distraction = _axes(data)
        segments = (Segment(0.0, float("inf"), drowsiness, distraction),)
    else:
        raise ValueError("manifest に segments も軸ラベルもありません。")

    return ClipManifest(video=video, subject=subject, context=context, segments=segments)


def load_manifest(path: str | Path) -> ClipManifest:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest が見つかりません: {p}")
    with p.open(encoding="utf-8") as f:
        return from_dict(json.load(f))


def manifests_from(path: str | Path) -> Iterator[ClipManifest]:
    """どんなデータセットでも、manifest(JSON)の形にしてここへ渡す唯一の入口。

    path が .json 単体ならその1本、ディレクトリなら中の *.json すべて。
    データセット固有の変換は、この manifest を生成する側（核の外）に置く。
    """
    p = Path(path)
    if p.is_file():
        yield load_manifest(p)
        return
    if not p.is_dir():
        raise FileNotFoundError(f"manifest のパスが見つかりません: {p}")
    files = sorted(p.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"{p} に *.json がありません。")
    for f in files:
        yield load_manifest(f)
