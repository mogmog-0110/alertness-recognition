"""動画ファイルからの映像入力。

カメラが無い環境での確認や、録画済み映像での評価に使う。
タイムスタンプはフレーム番号と fps から決めるので単調増加になる。
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import cv2

from ..contracts import Frame


def probe_video_fps(path: str | Path) -> float:
    """動画を開いて、利用可能な正の公称フレームレートを返す。"""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"動画が見つかりません: {source}")
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"動画を開けませんでした: {source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"動画のFPSを取得できませんでした: {source} (fps={fps!r})")
    return fps


class VideoFileSource:
    def __init__(self, path: str | Path) -> None:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"動画が見つかりません: {source}")
        self._cap = cv2.VideoCapture(str(source))
        if not self._cap.isOpened():
            raise RuntimeError(f"動画を開けませんでした: {source}")
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self._fps) or self._fps <= 0:
            self._cap.release()
            raise ValueError(
                f"動画のFPSを取得できませんでした: {source} (fps={self._fps!r})"
            )
        self._index = 0

    @property
    def fps(self) -> float:
        """動画コンテナから取得した検証済みの公称FPS。"""
        return self._fps

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
