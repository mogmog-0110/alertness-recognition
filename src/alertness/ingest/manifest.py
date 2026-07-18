"""取り込み用の正規化スキーマ。動画1本と、その時間区間ごとの軸別・段階ラベル。

正準ラベルは AXES の各軸 × 4段階（none/low/medium/high）。区間には「付いている軸だけ」を
持たせ、指定の無い軸は未アノテ（空）として扱う。これで片軸しか情報が無いデータ（眠気だけ、
ストレスだけ 等）を、他軸を none と誤って断定せずに取り込める。
外部データセットは配布形式がバラバラなので、まずこの共通形に落としてから特徴抽出に渡す。
配布形式→この形への変換は核の外で行い、核には manifest だけが渡る。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

LEVELS = ("none", "low", "medium", "high")
# 許容する軸名の語彙。ここに無い軸名は typo とみなして弾く。
AXES = ("drowsiness", "distraction", "concentration", "stress")


@dataclass(frozen=True)
class Segment:
    start: float  # 秒
    end: float  # 秒（この時刻は含まない）
    levels: Mapping[str, str]  # 軸名→段階（LEVELS）。付いている軸だけを持つ


@dataclass(frozen=True)
class ClipManifest:
    video: str
    subject: str
    context: str  # 用途（driving / study 等）。空でもよい
    segments: tuple[Segment, ...]

    def labels_at(self, timestamp: float) -> dict[str, str]:
        # 区間に当たる時刻は、その区間が持つ軸のラベルを返す。当たらない時刻は空（無ラベル）。
        for seg in self.segments:
            if seg.start <= timestamp < seg.end:
                return dict(seg.levels)
        return {}


def _level(value: object) -> str:
    v = str(value).lower()
    if v not in LEVELS:
        raise ValueError(f"レベルは {LEVELS} のいずれかである必要があります: {value}")
    return v


def _levels(data: dict) -> Mapping[str, str]:
    # data に含まれる軸だけを段階へ写す。1軸も無い区間は誤り。
    levels = {axis: _level(data[axis]) for axis in AXES if axis in data}
    if not levels:
        raise ValueError(f"軸ラベルがありません（{AXES} のいずれかが必要）: {data}")
    return MappingProxyType(levels)


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
            segments.append(Segment(start, end, _levels(s)))
        segments = tuple(segments)
    elif any(axis in data for axis in AXES):
        # 動画1本まるごと1ラベル（動画単位ラベルの公開データ向け）。
        segments = (Segment(0.0, float("inf"), _levels(data)),)
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
