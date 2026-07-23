"""各段の所要時間の計測。どこが遅いかを画面で確かめるための軽い仕組み。

「全体で何fps出ているか」だけでは、カメラ待ちなのか処理が重いのかが分からない。段ごとに
出せば切り分けられる。指数移動平均でならすので、表示がちらつかない。

計測対象はアプリのループという1つしかない資源なので、状態もモジュールに1つだけ持つ。
debug 表示が有効なときだけ働き、無効なら計測ごと止めて負荷をかけない。
"""

from __future__ import annotations

import time
from contextlib import contextmanager

_ALPHA = 0.1  # 指数移動平均の追従。小さいほど安定して見える
_ORDER = ("capture", "observe", "classify", "output")
_LABELS = {
    "capture": "cam",  # カメラからの取り込み待ち
    "observe": "detect",  # 検出＋特徴量＋rPPG
    "classify": "judge",  # cue と統合
    "output": "draw",  # 描画と表示
}

_times: dict[str, float] = {}
_enabled = False


def enable(on: bool) -> None:
    global _enabled
    _enabled = on
    if not on:
        _times.clear()


@contextmanager
def stage(name: str):
    if not _enabled:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - started) * 1000.0
        previous = _times.get(name)
        _times[name] = elapsed if previous is None else previous + _ALPHA * (elapsed - previous)


def summary() -> list[str]:
    """表示用の1行ずつ。計測していなければ空。"""
    if not _enabled or not _times:
        return []
    lines = [f"{_LABELS.get(k, k)}: {_times[k]:.1f}ms" for k in _ORDER if k in _times]
    total = sum(_times.values())
    if total > 0:
        lines.append(f"loop: {total:.1f}ms")
    return lines
