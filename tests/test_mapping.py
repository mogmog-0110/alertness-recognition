from __future__ import annotations

from alertness.ingest.manifest import load_manifest
from alertness.ingest.mapping import lookup, ordinal_bin, segment, write_manifest


def test_ordinal_bin_bins_by_thresholds():
    assert ordinal_bin(2, [4, 6, 8]) == "none"
    assert ordinal_bin(4, [4, 6, 8]) == "low"
    assert ordinal_bin(7, [4, 6, 8]) == "medium"
    assert ordinal_bin(9, [4, 6, 8]) == "high"


def test_lookup_uses_default_for_unknown():
    table = {"alert": "none", "drowsy": "high"}
    assert lookup("drowsy", table) == "high"
    assert lookup("unknown", table) == "none"


def test_write_manifest_roundtrips(tmp_path):
    segments = [segment(0, 10, drowsiness="high")]
    path = write_manifest(tmp_path / "m.json", "v.mp4", "s1", "driving", segments)

    m = load_manifest(path)
    assert m.context == "driving"
    assert m.labels_at(5.0) == {"drowsiness": "high", "distraction": "none"}
