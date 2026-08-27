"""ウィンドウの生成と表示。

表示の大きさは撮影解像度と切り離す。rPPG の精度は実効フレームレートで決まるので撮影は
軽いままにし、見づらいときはウィンドウ側を大きくする。拡大は OpenCV のウィンドウに任せる
（毎フレーム画像を作り直さずに済む）。

既定の AUTOSIZE ウィンドウは手で大きさを変えられないので、WINDOW_NORMAL で作る。
これで設定の大きさで開きつつ、後からドラッグでも変えられる。

表示・キャリブ・ガイド収録の3経路すべてがここを通る。以前は経路ごとに imshow していて、
ガイド収録中だけ設定が効かない状態になっていた。ウィンドウは1つしかない共有資源なので、
「もう大きさを決めたか」もモジュール側で1つだけ持つ。
"""

from __future__ import annotations

import cv2
import numpy as np

from .overlay import WINDOW_NAME

_sized: set[str] = set()


def show(image: np.ndarray, width: int = 0, window: str = WINDOW_NAME) -> None:
    """ウィンドウに出す。width>0 なら最初の1回だけその幅に合わせる（縦横比は保つ）。

    window を分けると、同じ判定を別々の画面に出せる（観客向けと運転者向け）。
    大きさを決めたかどうかはウィンドウごとに覚える。1つの旗で共有すると、
    先に開いた側だけが設定の大きさになり、もう一方は既定のまま開く。
    """
    if width > 0 and window not in _sized:
        _sized.add(window)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        height = max(1, round(image.shape[0] * width / image.shape[1]))
        cv2.resizeWindow(window, width, height)
    cv2.imshow(window, image)


def reset() -> None:
    """ウィンドウを閉じたあと、次に開くときまた大きさを合わせられるようにする。"""
    _sized.clear()
