"""PCカメラからの映像入力。

高フレームレートを狙うときは MJPG を指定する。多くの USB カメラは既定が非圧縮(YUY2)で、
USB 帯域が足りず 720p では 30fps 頭打ちになる。MJPG なら同じ帯域で 60fps 以上を出せる。
要求した設定が通るとは限らないので、実際に確保できた値を actual に持たせて確認できるようにする。

フレームの時刻には perf_counter を使う。Windows の monotonic は分解能が 15.6ms しかなく、
30fps（33ms間隔）だと時刻が飛び飛びの値に丸められる。この時刻は rPPG が実効サンプリング
周波数を出すのに使うので、丸めがそのまま心拍推定の誤差になる。perf_counter なら 0.0001ms。

取り込みは別スレッドで回す。read() は次のフレームがカメラから届くまで待つので、同じ
スレッドで処理まで済ませると「待ち時間 + 処理時間」が1周期になる。30fps のカメラで処理が
25ms なら 33+25=58ms、つまり 17fps しか出ない。取り込みを分ければ待ちと処理が重なるので、
処理時間だけが上限になる。スレッドは常に最新の1枚だけを保持し、古いフレームは捨てる
（判定は「いまの状態」を見るものなので、遅れて届く過去のフレームには価値がない）。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import cv2
import numpy as np

from ..contracts import Frame


class WebcamSource:
    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: float = 0.0,
        threaded: bool = True,
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
        self._threaded = threaded
        self._lock = threading.Lock()
        self._latest: tuple[int, np.ndarray, float] | None = None  # (連番, 画像, 時刻)
        self._grabbed = 0  # 取り込んだ枚数。既読と同じなら新しいフレームはまだ無い
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        if threaded:
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()

    @property
    def actual(self) -> tuple[int, int, float]:
        """カメラが実際に受け付けた (幅, 高さ, fps)。要求どおりとは限らない。"""
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(self._cap.get(cv2.CAP_PROP_FPS)),
        )

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            ok, image = self._cap.read()
            if not ok:
                break
            with self._lock:
                self._grabbed += 1
                self._latest = (self._grabbed, image, time.perf_counter())
        with self._lock:
            self._latest = None  # 取り込みが終わったことを frames() に伝える

    def frames(self) -> Iterator[Frame]:
        if not self._threaded:
            yield from self._frames_direct()
            return
        served = 0
        while True:
            with self._lock:
                latest = self._latest
                stopped = self._stop.is_set()
            if stopped or (
                latest is None and self._reader is not None and not self._reader.is_alive()
            ):
                break
            if latest is None or latest[0] == served:
                time.sleep(0.001)  # 次の1枚が届くまで少しだけ譲る
                continue
            served, image, captured = latest
            yield Frame(image=image, index=self._index, timestamp=captured, source_id="webcam")
            self._index += 1

    def _frames_direct(self) -> Iterator[Frame]:
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            yield Frame(
                image=image, index=self._index, timestamp=time.perf_counter(), source_id="webcam"
            )
            self._index += 1

    def close(self) -> None:
        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        self._cap.release()
