"""複数台対応の切り分けと、DeviceWorker の異常時の振る舞いのテスト。

RemoteHub 自体は test_remote_hub.py、Watchdog 自体は test_watchdog.py で確かめて
あるので、ここでは「複数台モードに入るかどうかの判断」と「1 台ぶんのループが
App._observe/_on_stall と同じ規則で動くか」だけを見る。
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from alertness.app import _MAX_DETECT_FAILURES, _wants_multi_device
from alertness.contracts import Frame
from alertness.multi_device import DeviceWorker


def _args(**overrides) -> argparse.Namespace:
    base = {"guided": False, "scenario": "", "video": None}
    base.update(overrides)
    return argparse.Namespace(**base)


def test_multi_device_needs_max_peers_above_one():
    config = {"source": {"type": "remote", "remote": {"max_peers": 2}}}
    assert _wants_multi_device(config, _args()) is True


def test_a_single_peer_config_stays_on_the_single_device_path():
    config = {"source": {"type": "remote", "remote": {"max_peers": 1}}}
    assert _wants_multi_device(config, _args()) is False


def test_webcam_never_uses_multi_device():
    config = {"source": {"type": "webcam"}}
    assert _wants_multi_device(config, _args()) is False


@pytest.mark.parametrize(
    "overrides", [{"guided": True}, {"scenario": "s.json"}, {"video": "a.mp4"}]
)
def test_single_actor_workflows_stay_off_multi_device(overrides):
    # ガイド収録・シナリオ再生・動画読み込みは 1 人を相手にする手順なので対象外。
    config = {"source": {"type": "remote", "remote": {"max_peers": 3}}}
    assert _wants_multi_device(config, _args(**overrides)) is False


class _Peer:
    def __init__(self) -> None:
        self.id = 1
        self.connected = True


class _Pipeline:
    """observe が指定回数だけ例外を投げるパイプラインの模擬。"""

    def __init__(self, failures: int) -> None:
        self.remaining = failures

    def observe(self, frame):
        if self.remaining > 0:
            self.remaining -= 1
            raise RuntimeError("検出器が落ちた")
        return frame


def _worker(pipeline) -> DeviceWorker:
    worker = DeviceWorker.__new__(DeviceWorker)  # __init__ は接続を要るので通さない
    worker._pipeline = pipeline
    worker._detect_failures = 0
    worker._peer_id = 1
    return worker


def _frame() -> Frame:
    return Frame(image=np.zeros((2, 2, 3), dtype=np.uint8), index=0, timestamp=0.0)


def test_a_single_detector_failure_is_skipped():
    worker = _worker(_Pipeline(failures=1))
    assert worker._observe(_frame()) is None
    assert worker._observe(_frame()) is not None


def test_a_permanently_broken_detector_is_given_up_on():
    worker = _worker(_Pipeline(failures=_MAX_DETECT_FAILURES + 5))
    for _ in range(_MAX_DETECT_FAILURES - 1):
        worker._observe(_frame())
    with pytest.raises(RuntimeError):
        worker._observe(_frame())


def test_a_stall_while_disconnected_is_not_reported(monkeypatch):
    # 端末の画面が自分で状態を示す（黄色表示）ので、台数が増えるほど埋もれる
    # コンソール出力は既定で出さない（log.detail = --verbose のときだけ）。
    import alertness.multi_device as multi_device

    calls: list[str] = []
    monkeypatch.setattr(multi_device.log, "detail", calls.append)
    worker = DeviceWorker.__new__(DeviceWorker)
    worker._peer = _Peer()
    worker._peer.connected = False
    worker._peer_id = 1
    worker._stall_reported = False
    worker._on_stall(5.0)
    assert worker._stall_reported is False
    assert calls == []


def test_recovery_is_only_reported_after_a_real_stall(monkeypatch):
    import alertness.multi_device as multi_device

    calls: list[str] = []
    monkeypatch.setattr(multi_device.log, "detail", calls.append)
    worker = DeviceWorker.__new__(DeviceWorker)
    worker._peer = _Peer()
    worker._peer_id = 1
    worker._stall_reported = False
    worker._on_recover(0.0)  # 一度も知らせていないので黙る
    assert calls == []
    worker._on_stall(5.0)
    worker._on_recover(0.0)
    assert any("とまっています" in c for c in calls)
    assert any("再開しました" in c for c in calls)


def test_recalibrate_builds_a_fresh_calibrator():
    # 使い切った calibrator をそのまま collect() し続けると、二度目の progress が
    # 即座に 1.0 のままになり「押しても一瞬で終わる」ように見える。
    config = {"calibration": {"duration_seconds": 3.0}, "camera": {"target_fps": 30}}
    worker = DeviceWorker.__new__(DeviceWorker)
    worker._config = config
    worker._pipeline = _Pipeline(failures=0)
    worker._pipeline.reset_state = lambda: None
    exhausted = object()
    worker._calibrator = exhausted

    worker._recalibrate()

    assert worker._calibrator is not exhausted
    assert worker._calibrator.progress == 0.0
    assert worker._calibrating is True


def test_remote_sink_options_falls_back_to_the_old_iphone_keys():
    # browser.yaml はまだ iphone_names で書かれている。ここが抜けると表示名の対応表が
    # 空のまま動き、内部名（英語）がそのまま画面に出て、アイコンの対応表からも外れる。
    from alertness.factory import remote_sink_options

    feedback = {"iphone_names": {"inattentive": "\u6ce8\u610f\u6563\u6f2b"}}
    options = remote_sink_options(feedback)
    assert options["names"] == {"inattentive": "\u6ce8\u610f\u6563\u6f2b"}


def test_remote_sink_options_prefers_the_new_key_when_both_are_set():
    from alertness.factory import remote_sink_options

    feedback = {
        "remote_names": {"inattentive": "new"},
        "iphone_names": {"inattentive": "old"},
    }
    assert remote_sink_options(feedback)["names"] == {"inattentive": "new"}


def test_device_worker_wires_names_through_remote_sink_options(monkeypatch):
    # DeviceWorker が重い pipeline/calibrator を作らずに済むよう factory を差し替える。
    import alertness.multi_device as multi_device

    monkeypatch.setattr(multi_device.factory, "build_pipeline", lambda config: object())
    monkeypatch.setattr(multi_device.factory, "build_calibrator", lambda config: object())
    config = {
        "feedback": {"iphone_names": {"inattentive": "\u6ce8\u610f\u6563\u6f2b"}},
    }
    worker = DeviceWorker(_Peer(), config)  # type: ignore[arg-type]
    assert worker._sink._names == {"inattentive": "\u6ce8\u610f\u6563\u6f2b"}
