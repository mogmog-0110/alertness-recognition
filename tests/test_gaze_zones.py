"""視線ゾーンの分類と、ゾーンごとの注意残高の減り方のテスト。"""

from __future__ import annotations

import pytest
from _helpers import FakeHistory, make_observation

from alertness.classifier.cues._zones import Zone, ZoneMap
from alertness.classifier.cues.attention_buffer import AttentionBufferCue
from alertness.contracts import Features

CAR = ZoneMap(enabled=True, forward_yaw=12.0, forward_gaze=0.035)


def test_forward_needs_both_gaze_and_head():
    assert CAR.classify(0.0, 0.0, 0.0) is Zone.FORWARD
    assert CAR.classify(0.20, 0.0, 0.0) is not Zone.FORWARD  # 目だけ大きく外れている
    assert CAR.classify(0.0, 30.0, 0.0) is not Zone.FORWARD  # 頭が横を向いている


def test_side_glance_is_a_mirror_check():
    assert CAR.classify(0.10, 25.0, 0.0) is Zone.MIRROR


def test_modest_downward_glance_is_the_instrument_cluster():
    assert CAR.classify(0.0, 5.0, 15.0) is Zone.INSTRUMENT


def test_steep_downward_glance_is_not_the_instrument_cluster():
    # 膝元のスマホはメーターと同じ「下向き＋正面寄り」になる。角度の上限で分ける。
    # 上限が無いと、最も危険な脇見が最も安全なゾーンに化ける。
    assert CAR.classify(0.0, 5.0, 40.0) is Zone.AWAY


def test_zones_are_off_until_the_angles_are_measured():
    # ミラーとメーターの角度は車種と取り付けで変わる。測る前に既定値を当てると
    # 「脇見をミラー確認と読む」ことになり、区別しないより危険になる。
    desk = ZoneMap()
    assert desk.classify(0.10, 25.0, 0.0) is Zone.AWAY
    assert desk.classify(0.0, 5.0, 15.0) is Zone.AWAY


def _run(cue, series):
    """(時刻, gaze_dx, yaw, pitch) の並びを順に食わせ、最後の結果を返す。"""
    frames = [
        Features({"gaze_dx": dx, "gaze_off": abs(dx), "yaw_rel": yaw, "pitch_rel": pitch}, t)
        for t, dx, yaw, pitch in series
    ]
    result = None
    for i in range(len(frames)):
        result = cue.evaluate(make_observation(frames[i], FakeHistory(frames[: i + 1])))
    return result


def _glance(t0: float, seconds: float, dx: float, yaw: float, pitch: float = 0.0):
    n = int(seconds / 0.1)
    return [(t0 + i * 0.1, dx, yaw, pitch) for i in range(n)]


def _car_cue(**kwargs):
    return AttentionBufferCue(capacity_seconds=2.0, zones={"enabled": True}, **kwargs)


def test_a_normal_mirror_check_barely_costs_anything():
    # 0.7秒のミラー確認は安全確認そのもの。ここで残高を削ると、確認を怠る運転者ほど
    # 高得点になるという逆転が起きる。
    series = _glance(0.0, 3.0, 0.0, 0.0) + _glance(3.0, 0.7, 0.10, 25.0)
    result = _run(_car_cue(), series)
    assert result.score > 0.85
    assert "ミラー" in result.detail


def test_a_long_mirror_stare_still_empties_the_buffer():
    # 猶予を過ぎればミラーでも同じ速さで減る。見つめ続けるのは前方不注意に違いない。
    series = _glance(0.0, 3.0, 0.0, 0.0) + _glance(3.0, 4.0, 0.10, 25.0)
    result = _run(_car_cue(), series)
    assert result.score == 0.0
    assert result.active


def test_looking_away_drains_faster_than_checking_a_mirror():
    away = _glance(0.0, 3.0, 0.0, 0.0) + _glance(3.0, 1.5, 0.30, 60.0)
    mirror = _glance(0.0, 3.0, 0.0, 0.0) + _glance(3.0, 1.5, 0.10, 25.0)
    assert _run(_car_cue(), away).score < _run(_car_cue(), mirror).score


def test_repeated_short_glances_still_accumulate():
    # 視覚的時分割。1回ずつは短くても、前方に戻る時間が足りなければ残高は戻らない。
    series = []
    t = 0.0
    series += _glance(t, 2.0, 0.0, 0.0)
    t += 2.0
    for _ in range(8):
        series += _glance(t, 0.9, 0.10, 25.0)
        t += 0.9
        series += _glance(t, 0.2, 0.0, 0.0)
        t += 0.2
    assert _run(_car_cue(refill_rate=0.5), series).score < 0.5


def test_a_non_boolean_enabled_is_refused():
    # 設定は YAML から来るので型が保証されない。1 や "yes" を通すと、測っていない角度で
    # ミラー・メーターを区別し始める（区別しないより危険な側に倒れる）。
    for value in (1, "true", 0.0):
        with pytest.raises(TypeError, match="zones.enabled"):
            ZoneMap(enabled=value)  # type: ignore[arg-type]


def test_a_boolean_enabled_passes():
    assert ZoneMap(enabled=True).enabled is True
    assert ZoneMap().enabled is False
