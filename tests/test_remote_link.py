"""iPhone との WebSocket 接続のテスト。

プロトコルは端末側で確定しているので、ここで確かめるのは「決められた形の
メッセージを、決められた意味で扱えているか」。特に撮影時刻を端末の値のまま
使うこと（受信時刻で代用すると生理指標が通信の揺らぎで歪む）。
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

from alertness.sources.remote import RemoteLink, RemoteSource

_HEADER = struct.Struct("<d")


def _jpeg(width: int = 32, height: int = 24) -> bytes:
    ok, buffer = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return bytes(buffer)


def _message(captured: float, width: int = 32, height: int = 24) -> bytes:
    return _HEADER.pack(captured) + _jpeg(width, height)


class _Device:
    """端末役。別スレッドで接続を保ち、送受信をテスト側から同期的に呼べるようにする。"""

    def __init__(self, port: int) -> None:
        self._port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws = None
        self.replies: list[dict] = []
        self._ready = threading.Event()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)
        self._thread.start()
        assert self._ready.wait(5.0), "端末役が接続できませんでした"

    async def _main(self) -> None:
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed

        self._loop = asyncio.get_running_loop()
        async with connect(f"ws://127.0.0.1:{self._port}") as ws:
            self._ws = ws
            self._ready.set()
            try:
                async for message in ws:
                    self.replies.append(json.loads(message))
            except ConnectionClosed:
                pass

    def send(self, data) -> None:
        assert self._loop is not None and self._ws is not None
        asyncio.run_coroutine_threadsafe(self._ws.send(data), self._loop).result(5.0)

    def close(self) -> None:
        if self._loop is not None and self._ws is not None:
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
        self._thread.join(timeout=2.0)


@pytest.fixture
def link():
    made = RemoteLink("127.0.0.1", 0)  # 0 で空いている番号を借りる
    made.wait_ready()
    yield made
    made.close()


def _wait(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _arrived(link: RemoteLink):
    return lambda: link.take_newer_than(0) is not None


def test_the_capture_time_comes_from_the_device(link):
    # 受信時刻で代用してはいけない。まばたき率も rPPG も時間方向の量なので、
    # 通信が詰まった分だけ生理指標が歪む。
    device = _Device(link.port)
    device.send(_message(12.5, 64, 48))
    assert _wait(_arrived(link))
    frame = next(islice(RemoteSource(link).frames(), 1))
    device.close()
    assert frame.timestamp == 12.5
    assert frame.image.shape == (48, 64, 3)
    assert frame.source_id == "iphone"


def test_only_the_newest_frame_survives(link):
    # 端末は送信が詰まると古いフレームを捨てる。受け側も同じで、遅れて届いた
    # 過去のフレームを順番に処理すると、警告が実時間から離れていく。
    device = _Device(link.port)
    for captured in (1.0, 2.0, 3.0):
        device.send(_message(captured))
    assert _wait(lambda: (got := link.take_newer_than(0)) is not None and got[2] == 3.0)
    frames = list(islice(RemoteSource(link).frames(), 1))
    device.close()
    assert [f.timestamp for f in frames] == [3.0]


def test_broken_messages_do_not_break_the_connection(link):
    # 1 枚読めないことは異常ではない。ここで接続を落とすと、以降ずっと無警告になる。
    device = _Device(link.port)
    device.send("text")  # バイナリではない
    device.send(_HEADER.pack(1.0))  # 中身が無い
    device.send(_HEADER.pack(1.0) + b"not a jpeg")
    device.send(_message(9.0))
    assert _wait(_arrived(link))
    frame = next(islice(RemoteSource(link).frames(), 1))
    device.close()
    assert frame.timestamp == 9.0


def _stamped(link: RemoteLink, served: int) -> tuple[int, float]:
    """served の次の 1 枚が届くまで待ち、その連番と時刻を返す。"""
    assert _wait(lambda: link.take_newer_than(served) is not None)
    got = link.take_newer_than(served)
    assert got is not None
    return got[0], got[2]


def test_a_reconnect_does_not_send_the_clock_backwards(link):
    # 端末を繋ぎ直すと撮影時刻は 0 付近へ戻る。時刻が戻ったフレームを検出器へ渡すと
    # MediaPipe の detect_for_video が例外を投げ、以降のフレームも全部そこで落ちる。
    # 「待ち受けを続けます」と言いながら判定が二度と戻らなくなる。
    first = _Device(link.port)
    first.send(_message(5.0))
    served, before = _stamped(link, 0)
    first.close()
    assert _wait(lambda: not link.connected)

    second = _Device(link.port)
    second.send(_message(0.0))
    _, after = _stamped(link, served)
    second.close()
    assert after > before


def test_the_device_intervals_survive_a_reconnect(link):
    # 繋ぎ直しの段差を吸収しても、区間の中の間隔は端末の値のままでなければならない。
    # ここが崩れると、まばたき率もあくび間隔も rPPG も全部ずれる。
    first = _Device(link.port)
    first.send(_message(5.0))
    served, _ = _stamped(link, 0)
    first.close()
    assert _wait(lambda: not link.connected)

    second = _Device(link.port)
    second.send(_message(0.0))
    served, start = _stamped(link, served)
    second.send(_message(1.5))
    _, later = _stamped(link, served)
    second.close()
    assert later - start == pytest.approx(1.5)


def test_the_result_goes_back_over_the_same_connection(link):
    device = _Device(link.port)
    # 接続はサーバ側で登録されて初めて返送先になる。端末側が繋がった瞬間に送ると、
    # まだ宛先が無くて捨てられる。
    assert _wait(lambda: link.connected)
    link.send({"level": "high", "message": "眠気が強いです", "alert": True})
    assert _wait(lambda: device.replies)
    device.close()
    assert device.replies[0]["message"] == "眠気が強いです"
    assert device.replies[0]["alert"] is True


def test_sending_without_a_device_is_harmless(link):
    link.send({"level": "none"})  # まだ誰も繋いでいない


def test_a_busy_port_is_reported(link):
    # 「iPhone が繋がらない」という症状は原因が見えにくいので、起動時に分かるようにする。
    second = RemoteLink("127.0.0.1", link.port)
    with pytest.raises(RuntimeError, match="待ち受けられませんでした"):
        second.wait_ready()
    second.close()


def test_interrupt_ends_the_waiting(link):
    # 画面もキーも無い運転では SIGTERM で終わらせるしかないが、繋がっていない間は
    # フレームを待ち続けるので、待ちを解けないと終われない。
    source = RemoteSource(link)
    source.interrupt()
    assert list(source.frames()) == []


def _iphone_config(port: int) -> dict:
    return {
        "source": {"type": "iphone", "iphone": {"host": "127.0.0.1", "port": port}},
        "feedback": {"window": False, "remote_features": ["ear"]},
        "assessment": {"dimensions": []},
    }


def test_the_source_and_the_sink_share_one_connection():
    # 映像を受けるのと結果を返すのが同じ接続でないと、端末は結果を受け取れない。
    from alertness import factory
    from alertness.feedback.remote import RemoteSink
    from alertness.labeling import LabelState

    config = _iphone_config(0)
    source = factory.build_source(config)
    sinks = factory.build_sinks(config, False, LabelState(""), source=source)
    try:
        sent = [s for s in sinks._sinks if isinstance(s, RemoteSink)]
        assert len(sent) == 1
        assert sent[0]._link is source.link
        assert sent[0]._features == ("ear",)
    finally:
        source.close()


def test_a_webcam_run_gets_no_device_sink():
    from alertness import factory
    from alertness.labeling import LabelState

    config = {"feedback": {"window": False}, "assessment": {"dimensions": []}}
    sinks = factory.build_sinks(config, False, LabelState(""), source=None)
    assert sinks._sinks == []


def test_a_text_message_becomes_a_command() -> None:
    # 端末は PC の画面もキーボードも触れないので、基準の取り直しは
    # この経路でしか頼めない。
    link = RemoteLink(port=0)
    try:
        link._accept('{"command": "recalibrate"}')
        assert link.take_commands() == ["recalibrate"]
        # 読んだ分は消える。同じ命令を二度実行してはいけない。
        assert link.take_commands() == []
    finally:
        link.close()


def test_broken_commands_are_ignored() -> None:
    # 壊れたテキストで待ち受けごと落ちてはいけない。
    link = RemoteLink(port=0)
    try:
        for text in ("", "{", "[]", '{"no_command": 1}', '{"command": 3}'):
            link._accept(text)
        assert link.take_commands() == []
    finally:
        link.close()


def test_the_page_is_served_on_the_websocket_port(tmp_path) -> None:
    """ページと WebSocket を同じポートで出す。

    別ポートで配ると端末で承認する証明書が 2 つになり、片方だけ承認して
    繋がらないという分かりにくい失敗をする。
    """
    root = tmp_path / "web"
    root.mkdir()
    (root / "index.html").write_text("<p>hello</p>", encoding="utf-8")

    link = RemoteLink(port=0, web_root=str(root))
    try:
        link.wait_ready()
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{link.port}/") as res:
            assert res.status == 200
            assert b"hello" in res.read()
    finally:
        link.close()


def test_the_page_server_refuses_paths_outside_the_root(tmp_path) -> None:
    # ../ でリポジトリの中身を読み出せてはいけない。
    root = tmp_path / "web"
    root.mkdir()
    (root / "index.html").write_text("ok", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

    link = RemoteLink(port=0, web_root=str(root))
    try:
        link.wait_ready()
        import urllib.error
        import urllib.request

        try:
            urllib.request.urlopen(f"http://127.0.0.1:{link.port}/../secret.txt")
            raise AssertionError("読み出せてしまった")
        except urllib.error.HTTPError as error:
            assert error.code == 404
    finally:
        link.close()


def test_commands_survive_alongside_frames() -> None:
    # 端末は映像と命令を同じ接続で混ぜて送る。片方が他方を壊してはいけない。
    link = RemoteLink(port=0)
    try:
        header = struct.pack("<d", 1.0)
        import cv2
        import numpy as np

        image = np.zeros((8, 8, 3), np.uint8)
        ok, buf = cv2.imencode(".jpg", image)
        assert ok
        link._accept(header + buf.tobytes())
        link._accept('{"command": "recalibrate"}')
        link._accept(header + buf.tobytes())

        assert link.take_commands() == ["recalibrate"]
        assert link.take_newer_than(0) is not None, "命令を挟んでも映像は届く"
    finally:
        link.close()


def test_the_old_config_key_still_works() -> None:
    """iphone は旧称。既存の設定を壊さない。

    最初にネイティブアプリで作ったときの名残で、いまはブラウザからも繋がる。
    名前を変えたからといって、動いている設定が黙って効かなくなってはいけない。
    """
    from alertness import factory

    for key in ("remote", "iphone"):
        config = {"source": {"type": key, key: {"host": "127.0.0.1", "port": 0}}}
        source = factory.build_source(config)
        try:
            assert source.link.port > 0, f"{key} で待ち受けられていない"
        finally:
            source.close()
