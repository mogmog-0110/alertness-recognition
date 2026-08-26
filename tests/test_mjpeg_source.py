"""スマートフォンをカメラにする経路（MJPEG）のテスト。

本物の HTTP は使わず、ストリームの読み口だけを差し替える。
"""

from __future__ import annotations

import time
from itertools import islice

import cv2
import numpy as np
import pytest

from alertness.sources import mjpeg


def _jpeg(value: int) -> bytes:
    image = np.full((8, 8, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


class _FakeStream:
    """MJPEG ストリームの模擬。バイト列を少しずつ返す。"""

    def __init__(self, payload: bytes, chunk: int = 300) -> None:
        self._payload = payload
        self._chunk = chunk
        self._at = 0

    def read(self, _size):
        if self._at >= len(self._payload):
            time.sleep(0.005)  # 相手が黙っている状態
            return b""
        piece = self._payload[self._at : self._at + self._chunk]
        self._at += self._chunk
        return piece

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _multipart(count: int) -> bytes:
    parts = []
    for i in range(count):
        parts.append(b"--boundary\r\nContent-Type: image/jpeg\r\n\r\n")
        parts.append(_jpeg(20 + i * 10))
        parts.append(b"\r\n")
    return b"".join(parts)


def test_frames_are_decoded_from_a_multipart_stream(monkeypatch):
    monkeypatch.setattr(mjpeg, "urlopen", lambda *a, **k: _FakeStream(_multipart(5)))
    source = mjpeg.MjpegSource("http://phone/video")
    try:
        frames = list(islice(source.frames(), 3))
    finally:
        source.close()
    assert len(frames) == 3
    assert all(f.image.shape == (8, 8, 3) for f in frames)
    assert all(f.source_id == "mjpeg" for f in frames)


def test_timestamps_are_monotonic(monkeypatch):
    monkeypatch.setattr(mjpeg, "urlopen", lambda *a, **k: _FakeStream(_multipart(8)))
    source = mjpeg.MjpegSource("http://phone/video")
    try:
        stamps = [f.timestamp for f in islice(source.frames(), 4)]
    finally:
        source.close()
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)


def test_a_dropped_connection_is_retried(monkeypatch):
    calls = {"n": 0}

    def flaky(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("接続が切れた")
        return _FakeStream(_multipart(5))

    monkeypatch.setattr(mjpeg, "urlopen", flaky)
    monkeypatch.setattr(mjpeg, "_RECONNECT_MIN", 0.01)
    source = mjpeg.MjpegSource("http://phone/video")
    try:
        frames = list(islice(source.frames(), 2))
    finally:
        source.close()
    assert len(frames) == 2
    assert source.reconnects >= 1


def test_a_non_mjpeg_response_does_not_grow_without_bound(monkeypatch):
    # 区切りが見つからない応答（HTML のエラーページなど）を延々と溜め込まないこと。
    source = mjpeg.MjpegSource.__new__(mjpeg.MjpegSource)
    source._frames = mjpeg.LatestFrame()
    buffer = bytearray(b"x" * (mjpeg._MAX_BUFFER + 10))
    source._drain(buffer)
    assert len(buffer) == 0


def test_an_empty_url_is_rejected():
    with pytest.raises(ValueError, match="URL"):
        mjpeg.MjpegSource("")
