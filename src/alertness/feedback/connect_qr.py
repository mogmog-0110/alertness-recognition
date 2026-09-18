"""端末が繋いでくるまで、接続先 URL の QR を PC の画面に出す。

URL を端末で打ち込ませると、IP の打ち間違いと「繋がらない」の区別がつかない。
読み取るだけにしておけば、繋がらないときは網の側の問題だと切り分けられる。

判定用のウィンドウとは別に開き、繋がったら閉じる。判定用のウィンドウを出さない
構成（ブラウザ版の既定）でも出せるようにするため。
"""

from __future__ import annotations

import time

import cv2
import numpy as np

WINDOW_NAME = "connect"
_KEY_QUIT = ord("q")
_MODULE_PX = 8
_QUIET_MODULES = 4  # QR の規格が求める周囲の余白。削ると読み取れない端末がある
# 待ちの間は 1ms 刻みで呼ばれる。毎回 imshow すると CPU を無駄に食う。
_REFRESH_SECONDS = 0.1


def render(url: str) -> np.ndarray:
    """QR だけの 1 枚。"""
    modules = np.pad(cv2.QRCodeEncoder.create().encode(url), _QUIET_MODULES, constant_values=255)
    size = modules.shape[0] * _MODULE_PX
    code = cv2.resize(modules, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(code, cv2.COLOR_GRAY2BGR)


class ConnectQr:
    """待っている間だけ QR のウィンドウを出す。

    ×で閉じられたら、次に繋がって切れるまでは出し直さない。閉じた人は URL を
    別の手段で得ているので、すぐ開き直すと邪魔になるだけ。
    """

    def __init__(self, url: str) -> None:
        self._image = render(url)
        self._shown = False
        self._dismissed = False
        self._broken = False
        self._last_refresh = 0.0

    def waiting(self) -> bool:
        """待ちの間に繰り返し呼ぶ。'q' が押されたら False（待つのをやめる）。"""
        if self._dismissed or self._broken:
            return True
        now = time.monotonic()
        if self._shown and now - self._last_refresh < _REFRESH_SECONDS:
            return True
        self._last_refresh = now
        try:
            if self._shown and cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self._shown = False
                self._dismissed = True
                return True
            cv2.imshow(WINDOW_NAME, self._image)
            self._shown = True
            return (cv2.waitKey(1) & 0xFF) != _KEY_QUIT
        except cv2.error as error:
            # 画面の無い環境。QR が出せないだけで、URL を打てば繋がるので止めない。
            self._broken = True
            print(f"[remote] 接続用の QR を表示できませんでした（{error.err}）。")
            return True

    def hide(self) -> None:
        """繋がったら閉じる。次に切れたときはまた出す。"""
        self._dismissed = False
        if not self._shown:
            return
        self._shown = False
        try:
            cv2.destroyWindow(WINDOW_NAME)
            cv2.waitKey(1)  # これを回さないと Windows では窓が残る
        except cv2.error:
            pass  # 既に閉じられている
