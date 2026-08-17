from __future__ import annotations

import numpy as np

from alertness.contracts import Frame
from alertness.ingest.manifest import ClipManifest, Segment
from alertness.ingest.runner import _write_rows
from alertness.ingest.segment_label import SegmentLabelProvider
from alertness.sources.frame_rate import downsample_frames


class _Frame:
    def __init__(self, timestamp: float) -> None:
        self.timestamp = timestamp


class _Source:
    def __init__(self, times: list[float]) -> None:
        self._times = times

    def frames(self):
        for t in self._times:
            yield _Frame(t)


class _Pipeline:
    def observe(self, frame):
        return frame

    def classify(self, obs):
        return None


class _Sink:
    def __init__(self, provider: SegmentLabelProvider) -> None:
        self._provider = provider
        self.seen: list[dict[str, str]] = []

    def emit(self, obs, assessment) -> None:
        # 書き出す瞬間の軸別ラベルを記録して、時刻→ラベルの配布を確かめる。
        self.seen.append(dict(self._provider.levels))


def test_write_rows_propagates_axis_labels_per_frame():
    manifest = ClipManifest(
        "v.mp4",
        "s1",
        "study",
        (
            Segment(0.0, 1.0, {"drowsiness": "none", "distraction": "low"}),
            Segment(1.0, 2.0, {"drowsiness": "high", "distraction": "none"}),
        ),
    )
    provider = SegmentLabelProvider(manifest)
    sink = _Sink(provider)

    written = _write_rows(_Pipeline(), _Source([0.0, 0.5, 1.0, 1.5, 2.5]), provider, sink)

    assert written == 5
    assert sink.seen == [
        {"drowsiness": "none", "distraction": "low"},
        {"drowsiness": "none", "distraction": "low"},
        {"drowsiness": "high", "distraction": "none"},
        {"drowsiness": "high", "distraction": "none"},
        {},
    ]


def test_write_rows_labels_the_resampled_csv_timestamps():
    manifest = ClipManifest(
        "v.mp4",
        "s1",
        "study",
        (
            Segment(0.0, 0.1, {"drowsiness": "none"}),
            Segment(0.1, 0.2, {"drowsiness": "high"}),
        ),
    )
    frames = [
        Frame(np.zeros((1, 1, 3)), index, index / 30.0)
        for index in range(7)
    ]
    sampled = list(downsample_frames(frames, 30.0, 15.0))
    provider = SegmentLabelProvider(manifest)
    sink = _Sink(provider)

    written = _write_rows(_Pipeline(), _SourceFrameList(sampled), provider, sink)

    assert written == 4
    assert [frame.timestamp for frame in sampled] == [0.0, 1 / 15, 2 / 15, 3 / 15]
    assert sink.seen == [
        {"drowsiness": "none"},
        {"drowsiness": "none"},
        {"drowsiness": "high"},
        {},
    ]


class _SourceFrameList:
    def __init__(self, frames: list[Frame]) -> None:
        self._frames = frames

    def frames(self):
        yield from self._frames
