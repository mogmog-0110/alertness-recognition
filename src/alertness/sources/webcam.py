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

読み取りに失敗しても取り込みは終わらせず、開き直して再接続を試みる。車載では配線が
一瞬外れることがあり、そこで終了してしまうと以降ずっと無警告になる。装置が黙ることは
誤警告よりも危険なので、諦めずに待ち続け、止まっていること自体は Watchdog が知らせる。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import cv2

from ..contracts import Frame
from ._latest import LatestFrame

_RECONNECT_MIN = 0.5  # 再接続の最初の待ち時間（秒）
_RECONNECT_MAX = 5.0  # 再接続の待ち時間の上限（秒）


class WebcamSource:
    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: float = 0.0,
        threaded: bool = True,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap = self._open()

        self._frame_index = 0
        self._threaded = threaded
        self._frames = LatestFrame()
        self._reconnects = 0
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        if threaded:
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()

    def _open(self) -> cv2.VideoCapture:
        # Windows では DSHOW を使うと起動が速く安定する。
        cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"カメラ(index={self._index})を開けませんでした。接続と使用許可を確認してください。"
            )
        # 解像度より先に MJPG を要求する。順番を逆にすると解像度が戻される機種がある。
        if self._fps > 0:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        if self._width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._fps > 0:
            cap.set(cv2.CAP_PROP_FPS, self._fps)
        return cap

    @property
    def actual(self) -> tuple[int, int, float]:
        """カメラが実際に受け付けた (幅, 高さ, fps)。要求どおりとは限らない。"""
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(self._cap.get(cv2.CAP_PROP_FPS)),
        )

    @property
    def reconnects(self) -> int:
        """再接続した回数。配線や電源を疑う材料として表示・記録に使う。"""
        return self._reconnects

    def _read_loop(self) -> None:
        backoff = _RECONNECT_MIN
        while not self._stop.is_set():
            ok, image = self._cap.read()
            if ok:
                backoff = _RECONNECT_MIN
                self._frames.put(image, time.perf_counter())
                continue
            # 読めなかった。終了せず開き直す。ここで諦めると以降ずっと無警告になる。
            if self._stop.wait(backoff):
                break
            backoff = min(_RECONNECT_MAX, backoff * 2)
            self._reconnect()

    def _reconnect(self) -> None:
        try:
            self._cap.release()
            self._cap = self._open()
            self._reconnects += 1
        except (RuntimeError, cv2.error):
            # まだ挿さっていない。次の周回でまた試すので、ここでは何もしない。
            pass

    def frames(self) -> Iterator[Frame]:
        if not self._threaded:
            yield from self._frames_direct()
            return
        served = 0
        while not self._stop.is_set():
            latest = self._frames.take_newer_than(served)
            if latest is None:
                time.sleep(0.001)  # 次の1枚が届くまで少しだけ譲る
                continue
            served, image, captured = latest
            yield Frame(
                image=image, index=self._frame_index, timestamp=captured, source_id="webcam"
            )
            self._frame_index += 1

    def _frames_direct(self) -> Iterator[Frame]:
        # 非スレッド動作。こちらも読めなくなったら開き直す（振る舞いを揃える）。
        backoff = _RECONNECT_MIN
        while not self._stop.is_set():
            ok, image = self._cap.read()
            if not ok:
                if self._stop.wait(backoff):
                    break
                backoff = min(_RECONNECT_MAX, backoff * 2)
                self._reconnect()
                continue
            backoff = _RECONNECT_MIN
            yield Frame(
                image=image,
                index=self._frame_index,
                timestamp=time.perf_counter(),
                source_id="webcam",
            )
            self._frame_index += 1

    def close(self) -> None:
        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        self._cap.release()
