"""PCカメラからの映像入力。

高フレームレートを狙うときは MJPG を指定する。多くの USB カメラは既定が非圧縮(YUY2)で、
USB 帯域が足りず 720p では 30fps 頭打ちになる。MJPG なら同じ帯域で 60fps 以上を出せる。
要求した設定が通るとは限らないので、実際に確保できた値を actual に持たせて確認できるようにする。
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import cv2

from ..contracts import Frame


class WebcamSource:
    def __init__(
        self, index: int = 0, width: int = 1280, height: int = 720, fps: float = 0.0
    ) -> None:
        # Windows では DSHOW を使うと起動が速く安定する。
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"カメラ(index={index})を開けませんでした。接続と使用許可を確認してください。"
            )
        # 解像度より先に MJPG を要求する。順番を逆にすると解像度が戻される機種がある。
        if fps > 0:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._index = 0

    @property
    def actual(self) -> tuple[int, int, float]:
        """カメラが実際に受け付けた (幅, 高さ, fps)。要求どおりとは限らない。"""
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(self._cap.get(cv2.CAP_PROP_FPS)),
        )

    def frames(self) -> Iterator[Frame]:
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            yield Frame(
                image=image, index=self._index, timestamp=time.monotonic(), source_id="webcam"
            )
            self._index += 1

    def close(self) -> None:
        self._cap.release()
