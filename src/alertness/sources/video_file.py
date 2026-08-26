"""動画ファイルからの映像入力。

カメラが無い環境での確認や、録画済み映像での評価に使う。
タイムスタンプはフレーム番号と fps から決めるので単調増加になる。

realtime=True にすると、実時間に合わせて待ちながら流す。人に見せるとき用。
既定（False）は待たずに最速で流すので、採点や取り込みが短時間で終わる。判定に渡る時刻は
どちらもフレーム番号から作るので、速さを変えても判定結果は変わらない。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import cv2

from ..contracts import Frame


class VideoFileSource:
    def __init__(self, path: str, realtime: bool = False) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"動画が見つかりません: {path}")
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"動画を開けませんでした: {path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._index = 0
        self._realtime = realtime

    def frames(self) -> Iterator[Frame]:
        started = time.perf_counter()
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            timestamp = self._index / self._fps
            if self._realtime:
                # 処理が追いつかないときは待たない（負の待ちを sleep しない）。遅れは
                # そのまま残るが、先送りして帳尻を合わせようとすると早送りになって
                # かえって見づらい。
                behind = timestamp - (time.perf_counter() - started)
                if behind > 0:
                    time.sleep(behind)
            yield Frame(
                image=image,
                index=self._index,
                timestamp=timestamp,
                source_id="video",
            )
            self._index += 1

    def close(self) -> None:
        self._cap.release()
