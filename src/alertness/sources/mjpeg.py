"""HTTP の MJPEG ストリームから映像を取る。スマートフォンをカメラにする経路。

車に載せるのは PC ではなくスマートフォン、という構成のための入口。Android の
「IP Webcam」など、既製のアプリの多くが MJPEG over HTTP を出すので、アプリを自作せずに
今日から試せる。PC は開発機として手元に置き、映像だけを受け取る。

**時刻の限界を承知して使うこと。** フレームの時刻はここ（受信側）で入れるので、通信の
揺らぎがそのまま時刻の揺らぎになる。眠気・注意逸脱の判定は数百 ms の揺らぎに耐えるが、
rPPG の心拍・呼吸は実効サンプリング周波数の推定が狂うと精度が落ちる。生体指標まで
要るなら、撮影した瞬間の時刻をフレームに添えて送るアプリを自作するしかない
（そのときは Frame.timestamp にその値を入れるだけで、以降の層は無修正で動く）。

無線は車内で不安定なので、USB テザリングでスマートフォン経由の有線接続にするのが堅い。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from urllib.request import urlopen

import cv2
import numpy as np

from ..contracts import Frame
from ._latest import LatestFrame

_SOI = b"\xff\xd8"  # JPEG の開始
_EOI = b"\xff\xd9"  # JPEG の終わり
_CHUNK = 8192
_MAX_BUFFER = 8 * 1024 * 1024  # 区切りが見つからないまま溜め込む上限
_RECONNECT_MIN = 0.5
_RECONNECT_MAX = 5.0


class MjpegSource:
    def __init__(self, url: str, timeout: float = 5.0) -> None:
        if not url:
            raise ValueError(
                "MJPEG の URL が空です。source.url にストリームの URL を指定してください。"
            )
        self._url = url
        self._timeout = timeout
        self._frames = LatestFrame()
        self._index = 0
        self._reconnects = 0
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    @property
    def reconnects(self) -> int:
        return self._reconnects

    def _read_loop(self) -> None:
        backoff = _RECONNECT_MIN
        while not self._stop.is_set():
            try:
                self._pump()
                backoff = _RECONNECT_MIN
            except OSError:
                # 通信が切れた。車内では珍しくないので終了せず開き直す。
                # 黙ることは誤警告より危険で、止まっていること自体は Watchdog が知らせる。
                self._reconnects += 1
            if self._stop.wait(backoff):
                break
            backoff = min(_RECONNECT_MAX, backoff * 2)

    def _pump(self) -> None:
        """1本の接続から、切れるまでフレームを取り出し続ける。"""
        with urlopen(self._url, timeout=self._timeout) as stream:  # noqa: S310 - 設定で渡す URL
            buffer = bytearray()
            while not self._stop.is_set():
                chunk = stream.read(_CHUNK)
                if not chunk:
                    return  # 相手が閉じた。呼び出し側が開き直す
                buffer.extend(chunk)
                self._drain(buffer)

    def _drain(self, buffer: bytearray) -> None:
        """溜まったバイト列から、完成した JPEG を取り出す。"""
        while True:
            start = buffer.find(_SOI)
            end = buffer.find(_EOI, start + 2) if start >= 0 else -1
            if start < 0 or end < 0:
                break
            jpeg = bytes(buffer[start : end + 2])
            del buffer[: end + 2]
            image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                self._frames.put(image, time.perf_counter())
        if len(buffer) > _MAX_BUFFER:
            # 区切りが見つからない＝MJPEG ではない応答。溜め続けるとメモリを食い潰す。
            buffer.clear()

    def frames(self) -> Iterator[Frame]:
        served = 0
        while not self._stop.is_set():
            latest = self._frames.take_newer_than(served)
            if latest is None:
                time.sleep(0.001)
                continue
            served, image, captured = latest
            yield Frame(image=image, index=self._index, timestamp=captured, source_id="mjpeg")
            self._index += 1

    def close(self) -> None:
        self._stop.set()
        self._reader.join(timeout=2.0)
        self._frames.clear()
