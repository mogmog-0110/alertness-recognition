from __future__ import annotations

import json

import pytest

from alertness.ingest.manifest import from_dict, manifests_from


def test_from_dict_segments_per_axis():
    m = from_dict(
        {
            "video": "v.mp4",
            "subject": "s1",
            "context": "study",
            "segments": [{"start": 0, "end": 2, "drowsiness": "high", "distraction": "none"}],
        }
    )
    assert m.context == "study"
    assert m.labels_at(1.0) == {"drowsiness": "high", "distraction": "none"}
    # 区間外は完全に無ラベル（軸キー自体が無い）。
    assert m.labels_at(3.0) == {}


def test_from_dict_single_label_covers_whole_video():
    m = from_dict({"video": "v.mp4", "drowsiness": "low"})
    # 指定した軸だけを持つ。distraction は付けていないので現れない（none と断定しない）。
    assert m.labels_at(0.0) == {"drowsiness": "low"}
    assert m.labels_at(10_000.0)["drowsiness"] == "low"


def test_from_dict_keeps_only_labeled_axes():
    # 片軸だけのデータ（ストレスだけ）が、他軸を none と誤断定せず取り込める。
    m = from_dict({"video": "v.mp4", "segments": [{"start": 0, "end": 5, "stress": "high"}]})
    assert m.labels_at(2.0) == {"stress": "high"}


def test_from_dict_rejects_segment_without_axis():
    with pytest.raises(ValueError):
        from_dict({"video": "v.mp4", "segments": [{"start": 0, "end": 1}]})


def test_from_dict_rejects_reversed_segment():
    with pytest.raises(ValueError):
        from_dict({"video": "v.mp4", "segments": [{"start": 2, "end": 1, "drowsiness": "low"}]})


def test_from_dict_rejects_invalid_level():
    with pytest.raises(ValueError):
        from_dict({"video": "v.mp4", "segments": [{"start": 0, "end": 1, "drowsiness": "sleepy"}]})


def test_manifests_from_directory(tmp_path):
    d = tmp_path / "manifests"
    d.mkdir()
    (d / "a.json").write_text(
        json.dumps({"video": "a.mp4", "drowsiness": "none"}), encoding="utf-8"
    )
    (d / "b.json").write_text(
        json.dumps({"video": "b.mp4", "distraction": "high"}), encoding="utf-8"
    )
    got = list(manifests_from(d))
    assert len(got) == 2
    assert {m.video for m in got} == {"a.mp4", "b.mp4"}
