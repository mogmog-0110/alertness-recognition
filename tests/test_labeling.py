"""録画ラベルの実行時状態とキー対応のテスト。"""

from __future__ import annotations

from alertness.labeling import LabelState, key_label_map


def test_label_state_is_mutable():
    state = LabelState("awake")
    assert state.value == "awake"
    state.value = "drowsiness"
    assert state.value == "drowsiness"


def test_key_label_map_assigns_in_order():
    mapping = key_label_map(["drowsiness", "distraction"])
    assert mapping[ord("0")] == ""
    assert mapping[ord("1")] == "awake"
    assert mapping[ord("2")] == "drowsiness"
    assert mapping[ord("3")] == "distraction"


def test_a_hand_pressed_label_has_no_axis_levels():
    # 軸別ラベルを埋めるのはシナリオ再生・取り込みだけ。読む側が毎回持っているか
    # 確かめずに済むよう、空の形は基底が持つ。
    labels = LabelState("drowsiness")
    assert labels.levels == {}


def test_advancing_the_clock_does_nothing_for_a_hand_pressed_label():
    labels = LabelState("drowsiness")
    labels.apply(12.0)
    assert labels.value == "drowsiness"
    assert labels.levels == {}
