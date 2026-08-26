"""録画CSVからイベント単位の採点を作る。

events.py は区間どうしを突き合わせる純粋な計算だけを持つ。こちらが CSV を読んで
「正解の危険区間」と「判定が出した警告区間」に変換する係。

区間はセッションをまたがない。別々の収録を繋げて1本の時間軸として扱うと、収録の
切れ目が長い1つの区間に化けて、誤警告の数も遅延も意味を失う。
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence

from .events import Episode, EventScore, episodes_from_flags, score_events

# 正解ラベルのうち「危険」とみなす段階。low は本人も気づかない程度の兆候なので、
# 警告できなくても見逃しには数えない。
DEFAULT_DANGER_LEVELS = ("medium", "high")


def _sessions(paths: Sequence[str], axis: str) -> list[list[Mapping[str, str]]]:
    """セッションごとの行の並び。session_id が無ければファイル単位で1つとみなす。"""
    grouped: list[list[Mapping[str, str]]] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if _timestamp(r) is not None]
        by_session: dict[str, list[Mapping[str, str]]] = {}
        for row in rows:
            by_session.setdefault(row.get("session_id", path), []).append(row)
        grouped.extend(by_session.values())
    return [rows for rows in grouped if len(rows) >= 2]


def _timestamp(row: Mapping[str, str]) -> float | None:
    try:
        return float(row.get("timestamp", ""))
    except ValueError:
        return None


def _level(row: Mapping[str, str], axis: str) -> int:
    raw = row.get(f"dim_{axis}_level", "")
    if raw == "":
        return 0
    try:
        return int(float(raw))
    except ValueError:
        return 0


def score_axis_events(
    paths: Sequence[str],
    axis: str,
    min_level: int = 2,
    danger_levels: Sequence[str] = DEFAULT_DANGER_LEVELS,
) -> EventScore:
    """1本の軸について、検出遅延と時間あたりの誤警告を出す。

    min_level は「警告を出した」とみなす段階（既定 2＝MEDIUM）。
    danger_levels は正解ラベルのうち危険とみなす段階。
    """
    truth: list[Episode] = []
    alerts: list[Episode] = []
    total = 0.0
    danger = set(danger_levels)

    for rows in _sessions(paths, axis):
        times = [float(r["timestamp"]) for r in rows]
        total += times[-1] - times[0]
        truth += episodes_from_flags(
            times, [(r.get(f"label_{axis}") or "").strip() in danger for r in rows]
        )
        alerts += episodes_from_flags(times, [_level(r, axis) >= min_level for r in rows])

    return score_events(truth, alerts, total)


def format_axis_events(paths: Sequence[str], axes: Sequence[str], min_level: int = 2) -> str:
    """全軸ぶんのイベント採点をまとめた文字列。ラベルが無い軸は飛ばす。"""
    from .events import format_event_score

    blocks: list[str] = []
    for axis in axes:
        score = score_axis_events(paths, axis, min_level)
        if score.truth_count == 0 and score.false_alarms == 0:
            continue  # その軸のラベルも警告も無い＝評価対象外
        blocks.append(f"[{axis}]\n{format_event_score(score)}")
    if not blocks:
        return "イベント単位の採点に使える軸別ラベル（label_<軸>）がありません。"
    return "\n\n".join(blocks)
