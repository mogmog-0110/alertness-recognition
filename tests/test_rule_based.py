"""統合方針（RuleBasedPolicy）のテスト。"""

from __future__ import annotations

from _helpers import make_observation

from alertness.classifier.policies.rule_based import RuleBasedPolicy
from alertness.classifier.states import DimensionSpec
from alertness.contracts import CueResult, Features, Level

LEVELS = {"low": 0.3, "medium": 0.6, "high": 0.8}


def _policy(specs, weights):
    # hysteresis_frames=1 なら平滑なし（その場の値がそのまま出る）。
    return RuleBasedPolicy(specs, weights, hysteresis_frames=1)


def _obs():
    return make_observation(Features({}, 0.0))


def test_strong_active_cue_drives_high():
    spec = DimensionSpec("drowsiness", LEVELS, ("eye_closure", "blink"))
    policy = _policy([spec], {"eye_closure": 1.0, "blink": 1.0})
    cues = [
        CueResult("eye_closure", "drowsiness", 1.0, True, ""),
        CueResult("blink", "drowsiness", 0.0, False, ""),
    ]
    result = policy.decide(_obs(), cues)
    assert result.dimensions["drowsiness"].level == Level.HIGH


def test_no_signal_is_none():
    spec = DimensionSpec("drowsiness", LEVELS, ("eye_closure",))
    policy = _policy([spec], {"eye_closure": 1.0})
    result = policy.decide(_obs(), [CueResult("eye_closure", "drowsiness", 0.0, False, "")])
    assert result.dimensions["drowsiness"].level == Level.NONE


def test_dimensions_are_independent():
    specs = [
        DimensionSpec("drowsiness", LEVELS, ("eye_closure",)),
        DimensionSpec("distraction", LEVELS, ("gaze_off",)),
    ]
    policy = _policy(specs, {"eye_closure": 1.0, "gaze_off": 1.0})
    cues = [
        CueResult("eye_closure", "drowsiness", 1.0, True, ""),
        CueResult("gaze_off", "distraction", 1.0, True, ""),
    ]
    result = policy.decide(_obs(), cues)
    assert result.dimensions["drowsiness"].level == Level.HIGH
    assert result.dimensions["distraction"].level == Level.HIGH


def test_inverted_axis_alerts_when_score_is_low():
    # 集中は高いほど良い軸。集中していない（score 低）ときに警告が立つ。
    spec = DimensionSpec("concentration", LEVELS, ("attention_hold",), "low", "low_concentration")
    policy = _policy([spec], {"attention_hold": 1.0})

    idle = policy.decide(_obs(), [CueResult("attention_hold", "concentration", 0.0, False, "")])
    dim = idle.dimensions["concentration"]
    assert dim.score == 0.0  # 軸そのものの値は「集中していない」まま
    assert dim.alarm == 1.0  # 警告の強さは反転する
    assert dim.level == Level.HIGH
    assert dim.display_name == "low_concentration"

    focused = policy.decide(_obs(), [CueResult("attention_hold", "concentration", 1.0, True, "")])
    assert focused.dimensions["concentration"].level == Level.NONE


def test_normal_axis_keeps_score_as_alarm():
    spec = DimensionSpec("drowsiness", LEVELS, ("eye_closure",))
    policy = _policy([spec], {"eye_closure": 1.0})
    result = policy.decide(_obs(), [CueResult("eye_closure", "drowsiness", 0.9, True, "")])
    dim = result.dimensions["drowsiness"]
    assert dim.alert_score is None
    assert dim.alarm == dim.score
    assert dim.display_name == "drowsiness"


def test_contributing_lists_active_cues_only():
    spec = DimensionSpec("drowsiness", LEVELS, ("eye_closure", "blink"))
    policy = _policy([spec], {"eye_closure": 1.0, "blink": 1.0})
    cues = [
        CueResult("eye_closure", "drowsiness", 1.0, True, ""),
        CueResult("blink", "drowsiness", 0.0, False, ""),
    ]
    result = policy.decide(_obs(), cues)
    assert result.dimensions["drowsiness"].contributing == ("eye_closure",)


def test_weighted_combination_requires_agreement():
    # combine=weighted の軸は、片方の cue だけが強く出ても警告まで届かない。
    spec = DimensionSpec("stress", LEVELS, ("hr_elevation", "facial_tension"), combine="weighted")
    policy = _policy([spec], {"hr_elevation": 1.0, "facial_tension": 1.0})

    alone = policy.decide(
        _obs(),
        [
            CueResult("hr_elevation", "stress", 1.0, True, ""),
            CueResult("facial_tension", "stress", 0.0, False, ""),
        ],
    )
    assert alone.dimensions["stress"].level < Level.MEDIUM  # 心拍だけでは立たない

    both = policy.decide(
        _obs(),
        [
            CueResult("hr_elevation", "stress", 0.9, True, ""),
            CueResult("facial_tension", "stress", 0.8, False, ""),
        ],
    )
    assert both.dimensions["stress"].level >= Level.MEDIUM  # 揃えば立つ


def test_max_combination_still_fires_on_a_single_strong_cue():
    # 眠気のように単独で決定的な兆候がある軸は、これまでどおり max。
    spec = DimensionSpec("drowsiness", LEVELS, ("eye_closure", "blink"))
    policy = _policy([spec], {"eye_closure": 1.0, "blink": 1.0})
    result = policy.decide(
        _obs(),
        [
            CueResult("eye_closure", "drowsiness", 1.0, True, ""),
            CueResult("blink", "drowsiness", 0.0, False, ""),
        ],
    )
    assert result.dimensions["drowsiness"].level == Level.HIGH
