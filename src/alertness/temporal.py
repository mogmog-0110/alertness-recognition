"""特徴量の時系列バッファ。

History を満たし、cue（PERCLOS・瞬きなど）や将来の時系列モデルが過去フレームを
読む口になる。リングバッファなので append で古いものから捨てる。
判定結果は不変だが、この履歴自体は性質上ためていく状態を持つ。

DROZY変換 のLoD算出もここで行う。LoDは連続値のスコアを
median・hysteresis・短区間統合して、連続した秒区間へ圧縮する。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np

from .contracts import Features

LOD_LEVELS = ("none", "low", "medium", "high")


def _median_scores(scores: Sequence[float], window_samples: int) -> list[float]:
    if window_samples <= 1:
        return [float(value) for value in scores]
    if window_samples % 2 == 0:
        window_samples += 1
    radius = window_samples // 2
    return [
        float(np.median(scores[max(0, index - radius) : min(len(scores), index + radius + 1)]))
        for index in range(len(scores))
    ]


def _hysteresis_labels(
    scores: Sequence[float], thresholds: Sequence[float], margin: float
) -> list[str]:
    if len(thresholds) != 3:
        raise ValueError("thresholds は3値である必要があります")
    if not scores:
        return []
    current = sum(float(scores[0]) >= float(boundary) for boundary in thresholds)
    labels = [LOD_LEVELS[current]]
    for value in scores[1:]:
        score = float(value)
        if current < 3 and score >= float(thresholds[current]) + margin:
            current += 1
        elif current > 0 and score < float(thresholds[current - 1]) - margin:
            current -= 1
        labels.append(LOD_LEVELS[current])
    return labels


def _merge_short_runs(labels: list[str], scores: Sequence[float], min_samples: int) -> list[str]:
    output = list(labels)
    if min_samples <= 1:
        return output
    changed = True
    while changed:
        changed = False
        start = 0
        while start < len(output):
            end = start + 1
            while end < len(output) and output[end] == output[start]:
                end += 1
            if end - start < min_samples and (start > 0 or end < len(output)):
                candidates: list[tuple[float, str]] = []
                run_score = float(np.median(scores[start:end]))
                if start > 0:
                    candidates.append(
                        (abs(run_score - float(scores[start - 1])), output[start - 1])
                    )
                if end < len(output):
                    candidates.append((abs(run_score - float(scores[end])), output[end]))
                replacement = min(candidates, key=lambda item: item[0])[1]
                output[start:end] = [replacement] * (end - start)
                changed = True
                break
            start = end
    return output


def smooth_lod_segments(
    scores: Sequence[float],
    *,
    thresholds: Sequence[float],
    timestamps: Sequence[float] | None = None,
    duration_seconds: float | None = None,
    stride_seconds: float = 1.0,
    median_seconds: float = 5.0,
    hysteresis_margin: float = 5.0,
    min_duration_seconds: float = 5.0,
) -> list[dict[str, object]]:
    """CDSをmedian・hysteresis・短区間統合し、連続した秒区間へ圧縮する。"""
    if not scores:
        return []
    if stride_seconds <= 0:
        raise ValueError("stride_seconds は正の値である必要があります")
    if timestamps is not None and len(timestamps) != len(scores):
        raise ValueError("timestamps と scores の長さが一致しません")
    median_samples = max(1, int(round(median_seconds / stride_seconds)))
    filtered = _median_scores(scores, median_samples)
    labels = _hysteresis_labels(filtered, thresholds, hysteresis_margin)
    labels = _merge_short_runs(
        labels, filtered, max(1, int(round(min_duration_seconds / stride_seconds)))
    )
    times = (
        list(timestamps)
        if timestamps is not None
        else [index * stride_seconds for index in range(len(scores))]
    )
    boundaries = [0.0]
    boundaries.extend(
        (float(times[index - 1]) + float(times[index])) / 2.0 for index in range(1, len(times))
    )
    inferred_end = float(times[-1]) + stride_seconds / 2.0
    boundaries.append(float(duration_seconds) if duration_seconds is not None else inferred_end)
    boundaries = [max(0.0, value) for value in boundaries]

    segments: list[dict[str, object]] = []
    start_index = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start_index]:
            start = boundaries[start_index]
            end = boundaries[index]
            if end > start:
                segments.append({"start": start, "end": end, "label": labels[start_index]})
            start_index = index
    return segments


def smooth_labels(labels: Sequence[str], *, window: int = 3) -> list[str]:
    """近傍のラベルを多数決で平滑化する。"""
    if not labels:
        return []
    if window <= 1:
        return list(labels)

    smoothed: list[str] = []
    for index in range(len(labels)):
        start = max(0, index - window // 2)
        end = min(len(labels), start + window)
        start = max(0, end - window)
        window_labels = list(labels[start:end])
        counts: dict[str, int] = {}
        for value in window_labels:
            counts[value] = counts.get(value, 0) + 1
        smoothed.append(max(counts.items(), key=lambda item: (item[1], item[0]))[0])
    return smoothed


def compress_segments(labels: Sequence[str], *, min_duration: int = 2) -> list[dict[str, object]]:
    """連続する同一ラベルを区間へ圧縮する。"""
    if not labels:
        return []

    segments: list[dict[str, object]] = []
    current_label = labels[0]
    start = 0
    for index in range(1, len(labels)):
        if labels[index] != current_label:
            if index - start >= min_duration:
                segments.append({"start": start, "end": index, "label": current_label})
            current_label = labels[index]
            start = index
    if len(labels) - start >= min_duration:
        segments.append({"start": start, "end": len(labels), "label": current_label})
    return segments


def map_labels_to_video_segments(
    labels: Sequence[str],
    *,
    fps: float = 30.0,
    min_duration_seconds: float = 1.0,
) -> list[dict[str, object]]:
    """1秒ごとの LoD ラベルを動画 FPS に合わせた区間へ写す。"""
    if not labels:
        return []
    if fps <= 0:
        raise ValueError("fps は正の値である必要があります")

    expanded: list[str] = []
    for label in labels:
        frame_span = max(1, int(round(fps)))
        expanded.extend([label] * frame_span)

    segments: list[dict[str, object]] = []
    current_label = expanded[0]
    start_frame = 0
    for index in range(1, len(expanded)):
        if expanded[index] != current_label:
            if index - start_frame >= max(1, int(round(fps * min_duration_seconds))):
                segments.append(
                    {
                        "start": start_frame / fps,
                        "end": index / fps,
                        "label": current_label,
                    }
                )
            current_label = expanded[index]
            start_frame = index
    if len(expanded) - start_frame >= max(1, int(round(fps * min_duration_seconds))):
        segments.append(
            {"start": start_frame / fps, "end": len(expanded) / fps, "label": current_label}
        )
    return segments


def build_manifest_segments(
    labels: Sequence[str],
    *,
    fps: float = 30.0,
    min_duration_seconds: float = 1.0,
) -> list[dict[str, object]]:
    """LoD ラベルを最終 manifest 用の区間へ整形する。"""
    if not labels:
        return []
    mapped = map_labels_to_video_segments(
        labels, fps=fps, min_duration_seconds=min_duration_seconds
    )
    return [
        {
            "start": float(item["start"]),
            "end": float(item["end"]),
            "label": str(item["label"]),
        }
        for item in mapped
    ]


class TemporalContext:
    def __init__(self, max_seconds: float = 60.0, fps: float = 30.0) -> None:
        self._fps = fps
        self._buf: deque[Features] = deque(maxlen=max(1, int(max_seconds * fps)))

    @property
    def fps(self) -> float:
        return self._fps

    def append(self, features: Features) -> None:
        self._buf.append(features)

    def latest(self) -> Features | None:
        return self._buf[-1] if self._buf else None

    @property
    def measured_fps(self) -> float:
        """実際に流れているフレームレート。要求値ではなく達成値を見るために使う。"""
        if len(self._buf) < 2:
            return self._fps
        span = self._buf[-1].timestamp - self._buf[0].timestamp
        return (len(self._buf) - 1) / span if span > 0 else self._fps

    def recent(self, seconds: float) -> Sequence[Features]:
        # 末尾から seconds 秒ぶんを返す。時刻は単調増加なので、古い側は見ずに末尾から
        # 遡って打ち切る。高フレームレートだとバッファが大きくなり、毎フレーム全走査すると
        # cue の呼び出し回数ぶん効いてくるため。
        if not self._buf:
            return []
        cutoff = self._buf[-1].timestamp - seconds
        out: list[Features] = []
        for features in reversed(self._buf):
            if features.timestamp < cutoff:
                break
            out.append(features)
        out.reverse()
        return out
