from __future__ import annotations

from alertness.ingest.manifest import ClipManifest, Segment
from alertness.ingest.runner import _write_rows
from alertness.ingest.segment_label import SegmentLabelProvider


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
        self.seen: list[tuple[str, str]] = []

    def emit(self, obs, assessment) -> None:
        # 書き出す瞬間の軸別ラベルを記録して、時刻→ラベルの配布を確かめる。
        self.seen.append((self._provider.drowsiness, self._provider.distraction))


def test_write_rows_propagates_axis_labels_per_frame():
    manifest = ClipManifest(
        "v.mp4",
        "s1",
        "study",
        (Segment(0.0, 1.0, "none", "low"), Segment(1.0, 2.0, "high", "none")),
    )
    provider = SegmentLabelProvider(manifest)
    sink = _Sink(provider)

    written = _write_rows(_Pipeline(), _Source([0.0, 0.5, 1.0, 1.5, 2.5]), provider, sink)

    assert written == 5
    assert sink.seen == [
        ("none", "low"),
        ("none", "low"),
        ("high", "none"),
        ("high", "none"),
        ("", ""),
    ]
