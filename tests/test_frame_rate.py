"""動画データセット共通のFPS検査・ダウンサンプリングを検証する。"""

from __future__ import annotations

import numpy as np
import pytest

from alertness.contracts import Frame
from alertness.sources.frame_rate import downsample_frames, validate_downsample_fps
from alertness.sources.video_file import probe_video_fps


def _frames(fps: float, count: int) -> list[Frame]:
    return [
        Frame(np.full((1, 1, 3), index, dtype=np.uint8), index, index / fps, "test")
        for index in range(count)
    ]


def test_downsamples_30_fps_to_exact_15_fps_grid() -> None:
    sampled = list(downsample_frames(_frames(30.0, 7), 30.0, 15.0))

    assert [int(frame.image[0, 0, 0]) for frame in sampled] == [0, 2, 4, 6]
    assert [frame.index for frame in sampled] == [0, 1, 2, 3]
    assert [frame.timestamp for frame in sampled] == pytest.approx([0, 1 / 15, 2 / 15, 3 / 15])


def test_non_integer_ratio_uses_distinct_frames_on_exact_output_grid() -> None:
    sampled = list(downsample_frames(_frames(25.0, 10), 25.0, 15.0))
    source_indices = [int(frame.image[0, 0, 0]) for frame in sampled]

    assert source_indices == [0, 2, 4, 5, 7, 9]
    assert len(source_indices) == len(set(source_indices))
    assert [frame.timestamp for frame in sampled] == pytest.approx(
        [index / 15.0 for index in range(len(sampled))]
    )


def test_equal_fps_keeps_every_frame() -> None:
    sampled = list(downsample_frames(_frames(15.0, 5), 15.0, 15.0))

    assert [int(frame.image[0, 0, 0]) for frame in sampled] == list(range(5))


def test_probe_video_fps_returns_validated_metadata_and_releases(monkeypatch, tmp_path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()

    class Capture:
        released = False

        def isOpened(self) -> bool:
            return True

        def get(self, _property: int) -> float:
            return 15.0

        def release(self) -> None:
            self.released = True

    capture = Capture()
    monkeypatch.setattr(
        "alertness.sources.video_file.cv2.VideoCapture", lambda _path: capture
    )

    assert probe_video_fps(video) == 15.0
    assert capture.released is True


@pytest.mark.parametrize("target", [0.0, -1.0, float("nan"), float("inf"), 16.0])
def test_rejects_invalid_or_upsampling_target(target: float) -> None:
    with pytest.raises(ValueError):
        validate_downsample_fps(15.0, target)
