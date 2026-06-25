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
