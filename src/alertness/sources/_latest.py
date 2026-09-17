"""取り込みスレッドと判定ループの間で「最新の1枚」だけを受け渡す箱。

古いフレームは捨てる。判定は「いまの状態」を見るものなので、遅れて届いた過去のフレームを
順番に処理すると、遅れが取り戻せないまま溜まり続け、警告が実時間から離れていく。
取り込み元（USB カメラ・ネットワーク越しのスマートフォン）が変わっても、この受け渡しの
性質は同じなので、ここに一本化する。

時刻は取り込み側が入れる。判定側で now() を取ると、届くまでの揺らぎがそのまま時刻の
揺らぎになり、rPPG が実効サンプリング周波数を出せなくなる。
"""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestFrame(Generic[T]):
    """最新の1枚と、その連番を保持する。連番は「もう出したか」の判定に使う。

    中身は復号済みの画像に限らない。ネットワーク越しの入力は、上書きされて
    捨てられるフレームまで復号しないよう、圧縮されたまま預ける。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: tuple[int, T, float] | None = None  # (連番, 画像, 取り込み時刻)
        self._count = 0

    def put(self, image: T, captured: float) -> None:
        with self._lock:
            self._count += 1
            self._latest = (self._count, image, captured)

    def take_newer_than(self, served: int) -> tuple[int, T, float] | None:
        """served より新しい1枚。まだ無ければ None。"""
        with self._lock:
            latest = self._latest
        if latest is None or latest[0] == served:
            return None
        return latest

    def clear(self) -> None:
        with self._lock:
            self._latest = None
