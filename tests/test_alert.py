"""警告音の段階とエスカレーションのテスト。

実際に音を鳴らさずに済むよう、再生だけを差し替えて呼ばれ方を見る。
"""

from __future__ import annotations

import pytest

from alertness.contracts import Level
from alertness.feedback.alert import AudioAlert


class _Recorder(AudioAlert):
    """再生の代わりに (軸名, 段) を記録するだけの AudioAlert。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(enabled=False, **kwargs)
        # 音源の用意と OS 依存の再生を飛ばし、鳴らす判断だけを試験対象にする。
        self._enabled = True
        self._paths = {"drowsiness": {"notice": None, "warn": None}}
        self.played: list[tuple[str, Level]] = []
        self.now = 0.0

    def _play(self, name: str, level: Level) -> None:
        self.played.append((name, level))

    def trigger(self, name: str, level: Level) -> None:  # 時刻を差し替える
        import time

        original = time.monotonic
        time.monotonic = lambda: self.now
        try:
            super().trigger(name, level)
        finally:
            time.monotonic = original


def test_no_sound_below_medium():
    alert = _Recorder()
    for level in (Level.NONE, Level.LOW):
        alert.trigger("drowsiness", level)
    assert alert.played == []


def test_medium_keeps_a_steady_interval():
    alert = _Recorder(cooldown_seconds=5.0)
    alert.trigger("drowsiness", Level.MEDIUM)  # 立った直後は待たせない
    alert.now = 4.9
    alert.trigger("drowsiness", Level.MEDIUM)
    assert len(alert.played) == 1  # まだ間隔に達していない
    alert.now = 5.1
    alert.trigger("drowsiness", Level.MEDIUM)
    assert len(alert.played) == 2


def test_rising_to_high_sounds_immediately():
    alert = _Recorder(cooldown_seconds=5.0)
    alert.trigger("drowsiness", Level.MEDIUM)
    alert.now = 0.1
    alert.trigger("drowsiness", Level.HIGH)  # 段が上がったら間隔を待たない
    assert alert.played == [("drowsiness", Level.MEDIUM), ("drowsiness", Level.HIGH)]


def test_high_tightens_the_interval_while_ignored():
    alert = _Recorder(cooldown_seconds=5.0, min_interval_seconds=1.5, escalate_factor=0.7)
    gaps = []
    last = 0.0
    alert.trigger("drowsiness", Level.HIGH)
    for _ in range(4):
        # 次に鳴るまで時間を進め、その所要時間を測る。
        while len(alert.played) == len(gaps) + 1:
            alert.now += 0.05
            alert.trigger("drowsiness", Level.HIGH)
        gaps.append(alert.now - last)
        last = alert.now
    assert gaps == sorted(gaps, reverse=True)  # 鳴らすたびに詰まる
    assert gaps[-1] >= 1.5  # 下限は割らない


def test_calming_down_resets_the_escalation():
    alert = _Recorder(cooldown_seconds=5.0)
    alert.trigger("drowsiness", Level.HIGH)
    alert.now = 10.0
    alert.trigger("drowsiness", Level.HIGH)
    alert.trigger("drowsiness", Level.NONE)  # 収まった
    plays = len(alert.played)

    alert.now = 10.1
    alert.trigger("drowsiness", Level.HIGH)  # 次の警告は最初の1回として即鳴る
    assert len(alert.played) == plays + 1


def test_unknown_dimension_is_silent():
    alert = _Recorder()
    alert.trigger("stress", Level.HIGH)
    assert alert.played == []


@pytest.mark.parametrize("shape", ["notice", "warn"])
def test_tone_shapes_produce_a_wav(tmp_path, shape):
    from alertness.feedback.tone import make_chime_wav

    path = tmp_path / f"{shape}.wav"
    make_chime_wav(path, "drowsy", shape)
    assert path.stat().st_size > 0
