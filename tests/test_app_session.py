"""端末がページを開き直したときの判定ループの振る舞い。

前の人の基準のまま判定すると、構え直しの間に顔が外れただけで警告が鳴る。
開き直しを知らされたら基準を捨て、測り直しが始まるまで判定を出さない。
"""

from __future__ import annotations

from _helpers import make_observation

from alertness.app import _SESSION_CALIBRATE_FALLBACK_SECONDS, App
from alertness.contracts import Features


class _Pipeline:
    def __init__(self) -> None:
        self.resets = 0

    def reset_state(self) -> None:
        self.resets += 1


class _Sinks:
    def __init__(self) -> None:
        self.prepared: list[float] = []

    def preparing(self, obs) -> None:
        self.prepared.append(obs.features.timestamp)


class _Source:
    def __init__(self, commands: list[str], connected: bool = True) -> None:
        self.commands = commands
        self.connected = connected

    def take_commands(self) -> list[str]:
        pending, self.commands = self.commands, []
        return pending


def _app(commands: list[str]) -> App:
    app = App.__new__(App)  # __init__ はカメラと検出器を用意するので通さない
    app._config = {}
    app._pipeline = _Pipeline()
    app._sinks = _Sinks()
    app._source = _Source(commands)
    app._calibrating = True
    app._preparing = False
    app._preparing_since = None
    app._stall_reported = False
    return app


def _obs(t: float):
    return make_observation(Features(values={}, timestamp=t))


def test_a_reopened_page_drops_the_baseline_and_waits():
    app = _app(["new_session"])
    app._handle_remote_commands()
    assert app._pipeline.resets == 1
    assert app._preparing and not app._calibrating

    app._wait_for_calibration(_obs(10.0))
    assert app._sinks.prepared == [10.0]


def test_the_device_command_starts_calibration(monkeypatch):
    app = _app(["new_session", "recalibrate"])
    monkeypatch.setattr("alertness.app.factory.build_calibrator", lambda _config: object())
    app._handle_remote_commands()
    assert app._calibrating and not app._preparing


def test_calibration_starts_anyway_if_the_command_never_comes(monkeypatch):
    # 命令を送らない古いページでも、測らないまま黙り続けてはいけない。
    app = _app(["new_session"])
    monkeypatch.setattr("alertness.app.factory.build_calibrator", lambda _config: object())
    app._handle_remote_commands()
    app._wait_for_calibration(_obs(10.0))
    app._wait_for_calibration(_obs(10.0 + _SESSION_CALIBRATE_FALLBACK_SECONDS))
    assert app._calibrating and not app._preparing


def test_silence_while_no_device_is_connected_is_not_reported(capsys):
    # 端末が繋がっていないのは待ち受け中の正常な姿で、操作者が直すものではない。
    app = _app([])
    app._source.connected = False
    app._on_stall(8.0)
    app._on_recover(0.0)
    assert capsys.readouterr().out == ""

    app._source.connected = True
    app._on_stall(3.0)
    app._on_recover(0.0)
    out = capsys.readouterr().out
    assert "[異常]" in out and "[復帰]" in out
