"""PCカメラからの映像入力。"""

from __future__ import annotations

import time
from collections.abc import Iterator

import cv2

from ..contracts import Frame


class WebcamSource:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        # Windows では DSHOW を使うと起動が速く安定する。
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"カメラ(index={index})を開けませんでした。接続と使用許可を確認してください。"
            )
        self._index = 0

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
