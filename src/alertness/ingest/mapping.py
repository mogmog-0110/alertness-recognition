"""データセット固有の変換器を数行で書くための小道具。

生フォーマットの読み取りと「どの元ラベルをどの段階に写すか」の判断は、
データセットごとに人が書く（ここでは抽象化しない）。ここが用意するのは、
その写像を宣言的に書くための最小の部品と、manifest の書き出しだけ。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .manifest import AXES, LEVELS, from_dict


def ordinal_bin(value: float, thresholds: Sequence[float], levels: Sequence[str] = LEVELS) -> str:
    """連続値/順序値を段階へ。例: ordinal_bin(kss, [4, 6, 8]) は 4未満=none, 8以上=high。"""
    if len(thresholds) != len(levels) - 1:
        raise ValueError("thresholds は levels より1つ少ない必要があります。")
    for i, t in enumerate(thresholds):
        if value < t:
            return levels[i]
    return levels[-1]


def lookup(value: object, table: Mapping[object, str], default: str = "none") -> str:
    """元のクラス名→段階の対応表。表に無い値は default。"""
    return table.get(value, default)


def segment(start: float, end: float, **levels: str) -> dict:
    """区間ラベルを1つ作る。付けたい軸だけキーワードで渡す（例: drowsiness="high"）。

    渡さなかった軸は未アノテ（空）として扱われる。none を明示したいときだけ none を渡す。
    """
    unknown = set(levels) - set(AXES)
    if unknown:
        raise ValueError(f"未知の軸: {sorted(unknown)}（使えるのは {AXES}）")
    return {"start": start, "end": end, **levels}


def write_manifest(
    path: str | Path,
    video: str,
    subject: str,
    context: str,
    segments: Sequence[dict],
) -> Path:
    data = {"video": video, "subject": subject, "context": context, "segments": list(segments)}
    from_dict(data)  # 書き出す前に妥当性を検証しておく
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return p
