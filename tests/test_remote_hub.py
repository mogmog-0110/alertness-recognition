"""複数端末を同時に受け付ける待ち受けのテスト。

RemoteLink（1 台の枠）と重なる部分（メッセージの形・ping/pong・ページ配信）は
test_remote_link.py で確かめてあるので、ここでは複数台ならではの性質だけを見る:
台数の上限、Peer 同士が独立していること、切断の通知。
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from itertools import islice

import cv2
import numpy as np
import pytest

from alertness.sources.remote import RemoteSource
from alertness.sources.remote_hub import RemoteHub

_HEADER = struct.Struct("<d")


def _jpeg(width: int = 32, height: int = 24) -> bytes:
    ok, buffer = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return bytes(buffer)


def _message(captured: float) -> bytes:
    return _HEADER.pack(captured) + _jpeg()


class _Device:
    """端末役。別スレッドで接続を保つ。"""

    def __init__(self, port: int) -> None:
        self._port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws = None
        self.replies: list[dict] = []
        self.closed_code: int | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)
        self._thread.start()
        assert self._ready.wait(5.0), "端末役が接続できませんでした"

    async def _main(self) -> None:
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed

        self._loop = asyncio.get_running_loop()
        try:
            async with connect(f"ws://127.0.0.1:{self._port}") as ws:
                self._ws = ws
                self._ready.set()
                try:
                    async for message in ws:
                        self.replies.append(json.loads(message))
                except ConnectionClosed:
                    pass
                self.closed_code = ws.close_code
        except Exception:  # noqa: BLE001 - 拒否された接続もテスト側で扱う
            self._ready.set()

    def send(self, data) -> None:
        assert self._loop is not None and self._ws is not None
        asyncio.run_coroutine_threadsafe(self._ws.send(data), self._loop).result(5.0)

    def close(self) -> None:
        if self._loop is not None and self._ws is not None:
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
        self._thread.join(timeout=2.0)


def _wait(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def hub():
    events: list[tuple[str, int]] = []
    made = RemoteHub(
        "127.0.0.1",
        0,
        max_peers=2,
        on_connect=lambda peer: events.append(("connect", peer.id)),
        on_disconnect=lambda peer: events.append(("disconnect", peer.id)),
    )
    made.wait_ready()
    made.events = events  # type: ignore[attr-defined]
    yield made
    made.close()


def test_up_to_max_peers_can_connect(hub):
    first, second = _Device(hub.port), _Device(hub.port)
    assert _wait(lambda: hub.connected_count == 2)
    first.close()
    second.close()


def test_a_connection_past_the_limit_is_refused(hub):
    first, second = _Device(hub.port), _Device(hub.port)
    assert _wait(lambda: hub.connected_count == 2)
    third = _Device(hub.port)
    assert _wait(lambda: third.closed_code is not None)
    assert hub.connected_count == 2  # 3 台目は数に入らない
    first.close()
    second.close()


def test_disconnecting_frees_a_slot_for_the_next_device(hub):
    first = _Device(hub.port)
    assert _wait(lambda: hub.connected_count == 1)
    first.close()
    assert _wait(lambda: hub.connected_count == 0)
    second = _Device(hub.port)
    assert _wait(lambda: hub.connected_count == 1)
    second.close()


def test_each_peer_gets_its_own_frame_stream(hub):
    # 台ごとに独立した Peer/LatestFrame を持つ。片方の映像がもう片方に混ざらない。
    connected = []
    hub._on_connect = lambda peer: connected.append(peer)  # type: ignore[attr-defined]
    first_dev, second_dev = _Device(hub.port), _Device(hub.port)
    assert _wait(lambda: len(connected) == 2)
    first_peer, second_peer = connected
    first_dev.send(_message(1.0))
    second_dev.send(_message(2.0))
    assert _wait(lambda: first_peer.take_newer_than(0) is not None)
    assert _wait(lambda: second_peer.take_newer_than(0) is not None)
    _, _, first_captured = first_peer.take_newer_than(0)
    _, _, second_captured = second_peer.take_newer_than(0)
    assert first_captured == 1.0
    assert second_captured == 2.0
    first_dev.close()
    second_dev.close()


def test_connect_and_disconnect_are_reported(hub):
    device = _Device(hub.port)
    assert _wait(lambda: ("connect", 1) in hub.events)  # type: ignore[attr-defined]
    device.close()
    assert _wait(lambda: ("disconnect", 1) in hub.events)  # type: ignore[attr-defined]


def test_a_result_reaches_only_its_own_device(hub):
    connected = []
    hub._on_connect = lambda peer: connected.append(peer)  # type: ignore[attr-defined]
    first_dev, second_dev = _Device(hub.port), _Device(hub.port)
    assert _wait(lambda: len(connected) == 2)
    first_peer, _second_peer = connected
    first_peer.send({"message": "1台目向け"})
    assert _wait(lambda: first_dev.replies)
    assert not second_dev.replies
    first_dev.close()
    second_dev.close()


def test_a_heartbeat_still_works_per_peer(hub):
    device = _Device(hub.port)
    device.send(json.dumps({"type": "ping"}))
    assert _wait(lambda: device.replies)
    assert device.replies == [{"type": "pong"}]
    device.close()


def test_a_hello_message_is_ignored_without_crashing(hub):
    # Peer は 1 接続を寿命とするので、ページの開き直しを見分ける仕組みは要らない。
    device = _Device(hub.port)
    device.send(json.dumps({"type": "hello", "session": "abc"}))
    device.send(json.dumps({"type": "ping"}))
    assert _wait(lambda: device.replies)
    assert device.replies == [{"type": "pong"}]
    device.close()


def test_frames_flow_through_remote_source(hub):
    connected = []
    hub._on_connect = lambda peer: connected.append(peer)  # type: ignore[attr-defined]
    device = _Device(hub.port)
    assert _wait(lambda: len(connected) == 1)
    device.send(_message(7.5))
    frame = next(islice(RemoteSource(connected[0]).frames(), 1))
    assert frame.timestamp == 7.5
    device.close()


def test_url_is_empty_without_a_web_root():
    hub = RemoteHub("127.0.0.1", 0, max_peers=1)
    hub.wait_ready()
    try:
        assert hub.url == ""
    finally:
        hub.close()
