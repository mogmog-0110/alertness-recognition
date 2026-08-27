"""判定が流れ続けていることを見張る番人。

この装置の最悪の壊れ方は「警告を出しすぎる」ではなく「黙る」こと。カメラが外れても、
検出器が例外で止まっても、画面には最後の判定が残ったままになり、運転者からは
正常に動いているのと区別がつかない。黙っていることを検出できるのは、判定の流れそのものを
外から見ている側だけなので、ループとは別のスレッドで時計を持つ。

止まったことを知らせる先は呼び出し側に任せる（音を鳴らすか、ログに書くか、車両側の
表示に出すかは配置で変わる）。ここは「いつ知らせるか」だけを持つ。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class Watchdog:
    """一定時間 beat() が来なければ on_stall を呼ぶ。

    復帰したら on_recover を一度だけ呼ぶ。止まっている間は repeat_seconds ごとに
    on_stall を鳴らし直す（1回きりだと、鳴った瞬間に気づかなければ二度と伝わらない）。
    """

    def __init__(
        self,
        stall_seconds: float = 3.0,
        repeat_seconds: float = 5.0,
        on_stall: Callable[[float], None] | None = None,
        on_recover: Callable[[float], None] | None = None,
        check_interval: float = 0.5,
    ) -> None:
        self.stall_seconds = stall_seconds  # これだけ判定が来なければ異常とみなす
        self.repeat_seconds = repeat_seconds  # 異常が続く間、知らせ直す間隔
        self._on_stall = on_stall
        self._on_recover = on_recover
        self._check_interval = check_interval
        self._lock = threading.Lock()
        self._last_beat = time.monotonic()
        self._stalled = False
        self._last_notice = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def beat(self) -> None:
        """1フレーム処理できたことを伝える。判定ループから毎回呼ぶ。"""
        with self._lock:
            self._last_beat = time.monotonic()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="watchdog")
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self._check_interval):
            self._tick(time.monotonic())

    def _tick(self, now: float) -> None:
        """1回ぶんの判断。時計を渡す形にしてあるのは、テストで待たずに進めるため。"""
        with self._lock:
            silent = now - self._last_beat
        if silent < self.stall_seconds:
            if self._stalled:
                self._stalled = False
                self._notify(self._on_recover, silent)
            return
        if not self._stalled:
            self._stalled = True
            self._last_notice = now
            self._notify(self._on_stall, silent)
            return
        if now - self._last_notice >= self.repeat_seconds:
            self._last_notice = now
            self._notify(self._on_stall, silent)

    @staticmethod
    def _notify(callback: Callable[[float], None] | None, silent: float) -> None:
        if callback is None:
            return
        try:
            callback(silent)
        except Exception as error:  # noqa: BLE001 - 通知の失敗で監視まで止めない
            print(f"[watchdog] 通知に失敗しました: {type(error).__name__}")
