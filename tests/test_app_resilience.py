"""判定ループが単発の故障で止まらないことのテスト。

車載ではセンサ断＝無警告なので、1フレームの失敗で終了してはいけない。
逆に壊れ続けているのに回り続けても意味がないので、見切る条件も確かめる。
"""

from __future__ import annotations

import numpy as np
import pytest

from alertness.app import _MAX_DETECT_FAILURES, App
from alertness.contracts import Frame


class _Pipeline:
    """observe が指定回数だけ例外を投げるパイプラインの模擬。"""

    def __init__(self, failures: int) -> None:
        self.remaining = failures
        self.observed = 0

    def observe(self, frame):
        if self.remaining > 0:
            self.remaining -= 1
            raise RuntimeError("検出器が落ちた")
        self.observed += 1
        return frame


def _app(pipeline) -> App:
    app = App.__new__(App)  # __init__ はカメラを開くので通さない
    app._pipeline = pipeline
    app._detect_failures = 0
    return app


def _frame() -> Frame:
    return Frame(image=np.zeros((2, 2, 3), dtype=np.uint8), index=0, timestamp=0.0)


def test_a_single_detector_failure_is_skipped(capsys):
    app = _app(_Pipeline(failures=1))
    assert app._observe(_frame()) is None  # そのフレームは捨てる
    assert app._observe(_frame()) is not None  # 次は通る
    assert "失敗" in capsys.readouterr().out


def test_the_failure_counter_resets_after_a_good_frame():
    pipeline = _Pipeline(failures=1)
    app = _app(pipeline)
    app._observe(_frame())
    app._observe(_frame())
    assert app._detect_failures == 0


def test_a_permanently_broken_detector_is_given_up_on():
    # 壊れ続けているのに回り続けても、無警告のまま動いているふりをするだけ。
    app = _app(_Pipeline(failures=_MAX_DETECT_FAILURES + 5))
    for _ in range(_MAX_DETECT_FAILURES - 1):
        app._observe(_frame())
    with pytest.raises(RuntimeError):
        app._observe(_frame())


class _WaitingSource:
    """繋がるまでフレームを待つ入力源の代用。"""

    def __init__(self) -> None:
        self.interrupted = False

    def interrupt(self) -> None:
        self.interrupted = True


def test_request_stop_ends_the_loop_without_a_window():
    # 画面もキーも無い運転では、これが唯一の止め方になる。
    app = App.__new__(App)
    app._stopping = False
    app._source = _WaitingSource()
    app.request_stop()
    assert app._stopping


def test_request_stop_also_releases_a_waiting_source():
    # ネットワーク越しの入力は繋がっていない間フレームを待つ。旗を立てるだけでは、
    # 待ちの中にいるループが終了の合図を読めない。
    app = App.__new__(App)
    app._stopping = False
    app._source = _WaitingSource()
    app.request_stop()
    assert app._source.interrupted


def test_request_stop_works_for_a_source_without_interrupt():
    app = App.__new__(App)
    app._stopping = False
    app._source = object()  # webcam など、待ちを解く必要が無い入力
    app.request_stop()
    assert app._stopping
