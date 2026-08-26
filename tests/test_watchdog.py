"""判定が止まったことを知らせる番人のテスト。

時計を渡す _tick を直接叩く。実時間を待つと、テストが遅いうえに取りこぼしで揺れる。
"""

from __future__ import annotations

from alertness.watchdog import Watchdog


def _watchdog(**kwargs):
    stalls: list[float] = []
    recovers: list[float] = []
    dog = Watchdog(on_stall=stalls.append, on_recover=recovers.append, **kwargs)
    return dog, stalls, recovers


def test_silence_shorter_than_the_limit_is_not_reported():
    dog, stalls, _ = _watchdog(stall_seconds=3.0)
    dog._last_beat = 0.0
    dog._tick(2.9)
    assert stalls == []


def test_silence_past_the_limit_is_reported_once():
    dog, stalls, _ = _watchdog(stall_seconds=3.0, repeat_seconds=5.0)
    dog._last_beat = 0.0
    dog._tick(3.1)
    dog._tick(3.2)  # まだ知らせ直す間隔ではない
    assert len(stalls) == 1


def test_a_continuing_stall_is_reported_again():
    # 1回きりだと、鳴った瞬間に気づかなければ二度と伝わらない。
    dog, stalls, _ = _watchdog(stall_seconds=3.0, repeat_seconds=5.0)
    dog._last_beat = 0.0
    dog._tick(3.1)
    dog._tick(8.2)
    assert len(stalls) == 2


def test_recovery_is_reported_once():
    dog, stalls, recovers = _watchdog(stall_seconds=3.0)
    dog._last_beat = 0.0
    dog._tick(3.1)
    dog.beat()
    dog._last_beat = 10.0
    dog._tick(10.1)
    dog._tick(10.2)
    assert len(recovers) == 1
    assert len(stalls) == 1


def test_a_failing_notifier_does_not_stop_the_watchdog():
    # 通知が壊れても監視は続く。ここで例外が漏れると、番人ごと死ぬ。
    def explode(_seconds):
        raise RuntimeError("通知先が落ちている")

    dog = Watchdog(stall_seconds=3.0, repeat_seconds=1.0, on_stall=explode)
    dog._last_beat = 0.0
    dog._tick(3.1)
    dog._tick(4.2)  # 2回目も呼ばれる＝監視が生きている
    assert dog._stalled


def test_start_and_close_are_safe_to_repeat():
    dog = Watchdog(check_interval=0.01)
    dog.start()
    dog.start()  # 二重に立てない
    dog.close()
    dog.close()
