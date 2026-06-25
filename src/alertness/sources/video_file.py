"""動画ファイルからの映像入力。

カメラが無い環境での確認や、録画済み映像での評価に使う。
タイムスタンプはフレーム番号と fps から決めるので単調増加になる。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from ..contracts import Frame


class VideoFileSource:
    def __init__(self, path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"動画が見つかりません: {path}")
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"動画を開けませんでした: {path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._index = 0

    def frames(self) -> Iterator[Frame]:
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            yield Frame(
                image=image,
                index=self._index,
                timestamp=self._index / self._fps,
                source_id="video",
            )
            self._index += 1

    def close(self) -> None:
        self._cap.release()
