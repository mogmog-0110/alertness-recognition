"""取り込みスレッドの振る舞いのテスト（カメラ無しで確認できる部分）。"""

from __future__ import annotations

import time

import numpy as np
import pytest

from alertness.sources import webcam


class _FakeCapture:
    """指定間隔でしかフレームを返さないカメラの模擬。read() は待つ。"""

    def __init__(self, interval: float = 0.033, limit: int = 20) -> None:
        self.interval = interval
        self.limit = limit
        self.reads = 0
        self.released = False

    def isOpened(self):  # noqa: N802 (cv2 の名前に合わせる)
        return True

    def set(self, *_args):
        return True

    def get(self, *_args):
        return 0.0

    def read(self):
        if self.reads >= self.limit:
            return False, None
        self.reads += 1
        time.sleep(self.interval)
        return True, np.full((4, 4, 3), self.reads % 255, dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def fake_capture(monkeypatch):
    created = {}

    def factory(interval=0.02, limit=20):
        capture = _FakeCapture(interval, limit)
        created["capture"] = capture
        monkeypatch.setattr(webcam.cv2, "VideoCapture", lambda *a: capture)
        return capture

    return factory


def test_threaded_source_does_not_wait_for_the_camera(fake_capture):
    # 取り込みが別スレッドなら、処理中も裏で次のフレームが貯まる。
    fake_capture(interval=0.02)
    source = webcam.WebcamSource(threaded=True)
    try:
        time.sleep(0.1)  # 裏で数枚取り込まれる
        started = time.perf_counter()
        frame = next(source.frames())
        assert (time.perf_counter() - started) < 0.015  # 待たずに受け取れる
        assert frame.image.shape == (4, 4, 3)
    finally:
        source.close()


def test_threaded_source_never_serves_the_same_frame_twice(fake_capture):
    # 同じフレームを2回出すと、時刻が重複して rPPG の推定が壊れる。
    fake_capture(interval=0.01, limit=6)
    source = webcam.WebcamSource(threaded=True)
    try:
        stamps = [f.timestamp for f, _ in zip(source.frames(), range(5), strict=False)]
    finally:
        source.close()
    assert len(set(stamps)) == len(stamps)
    assert stamps == sorted(stamps)  # 時刻は単調増加


def test_direct_mode_still_works(fake_capture):
    capture = fake_capture(interval=0.001, limit=3)
    source = webcam.WebcamSource(threaded=False)
    frames = list(source.frames())
    source.close()
    assert len(frames) == 3
    assert capture.released


def test_frame_timestamps_use_a_high_resolution_clock(fake_capture):
    """Windows の monotonic は 15.6ms 刻み。丸められた時刻は rPPG の実効fps推定を壊す。"""
    fake_capture(interval=0.005, limit=30)
    source = webcam.WebcamSource(threaded=True)
    try:
        stamps = [f.timestamp for f, _ in zip(source.frames(), range(10), strict=False)]
    finally:
        source.close()
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    assert all(g > 0 for g in gaps)  # 同じ時刻のフレームが出ない
    assert min(gaps) < 0.015  # 15.6ms より細かい間隔を表現できている
