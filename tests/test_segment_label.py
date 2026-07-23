from __future__ import annotations

from alertness.ingest.manifest import ClipManifest, Segment
from alertness.ingest.segment_label import SegmentLabelProvider


def test_provider_resolves_axis_labels_by_time():
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

    provider.apply(0.5)
    assert provider.levels == {"drowsiness": "none", "distraction": "low"}
    provider.apply(1.5)
    assert provider.levels == {"drowsiness": "high", "distraction": "none"}
    provider.apply(9.0)
    assert provider.levels == {}
