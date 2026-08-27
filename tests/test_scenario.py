"""シナリオ再生（動画＋期待ラベルを実時間で流す）のテスト。"""

from __future__ import annotations

import json
import time
from itertools import islice

import cv2
import numpy as np
import pytest

from alertness.app import App
from alertness.ingest.segment_label import SegmentLabelProvider
from alertness.sources.video_file import VideoFileSource


def _video(path, seconds=1.0, fps=30.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (32, 24))
    for _ in range(int(seconds * fps)):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()
    return str(path)


def _manifest(tmp_path, video):
    data = {
        "video": video,
        "subject": "demo",
        "segments": [
            {"start": 0.0, "end": 0.5, "drowsiness": "none"},
            {"start": 0.5, "end": 10.0, "drowsiness": "high"},
        ],
    }
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_realtime_playback_paces_itself(tmp_path):
    # 舞台に出すときは実時間で流す。最速で流すと、判定より先に映像が終わってしまう。
    path = _video(tmp_path / "clip.mp4", seconds=0.5)
    started = time.perf_counter()
    source = VideoFileSource(path, realtime=True)
    frames = list(islice(source.frames(), 10))
    elapsed = time.perf_counter() - started
    source.close()
    assert len(frames) == 10
    assert elapsed > 0.25  # 10 フレーム分 (0.33 秒) 近く待っている


def test_fast_playback_is_still_the_default(tmp_path):
    # 採点や取り込みは待たずに済ませたい。判定に渡る時刻は同じなので結果は変わらない。
    path = _video(tmp_path / "clip.mp4", seconds=1.0)
    source = VideoFileSource(path)
    started = time.perf_counter()
    frames = list(islice(source.frames(), 20))
    elapsed = time.perf_counter() - started
    source.close()
    assert len(frames) == 20
    assert elapsed < 0.2


def test_timestamps_do_not_depend_on_playback_speed(tmp_path):
    path = _video(tmp_path / "clip.mp4", seconds=1.0)
    fast = [f.timestamp for f in islice(VideoFileSource(path).frames(), 5)]
    slow = [f.timestamp for f in islice(VideoFileSource(path, realtime=True).frames(), 5)]
    assert fast == slow


def test_expected_labels_follow_the_clock(tmp_path):
    from alertness.ingest.manifest import load_manifest

    video = _video(tmp_path / "clip.mp4")
    provider = SegmentLabelProvider(load_manifest(_manifest(tmp_path, video)))

    provider.apply(0.2)
    assert provider.levels == {"drowsiness": "none"}
    provider.apply(3.0)
    assert provider.levels == {"drowsiness": "high"}
    provider.apply(50.0)
    assert provider.levels == {}  # 区間の外は未アノテ


def test_scenario_selects_the_video_from_the_manifest(tmp_path):
    video = _video(tmp_path / "clip.mp4")
    app = App.__new__(App)
    app._scenario = App._load_scenario(_manifest(tmp_path, video))
    assert app._scenario.video == video
    labels = app._make_labels("")
    assert isinstance(labels, SegmentLabelProvider)


def test_a_missing_scenario_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        App._load_scenario(str(tmp_path / "missing.json"))
