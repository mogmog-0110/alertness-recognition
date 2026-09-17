"""統合方針（RuleBasedPolicy）のテスト。"""

from __future__ import annotations

import math

import pytest
from _helpers import make_observation

from alertness.classifier.policies.rule_based import RuleBasedPolicy
from alertness.classifier.states import DimensionSpec
from alertness.contracts import CueResult, Features, Level

LEVELS = {"low": 0.3, "medium": 0.6, "high": 0.8}


def _policy(specs, weights):
    # attack/release ともに 1 なら平滑なし（その場の値がそのまま出る）。
    return RuleBasedPolicy(specs, weights, attack_frames=1, release_frames=1)


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


def test_alarm_rises_faster_than_it_falls():
    # 安全側の装置なので、危険は速く出し、解除はゆっくりにする。
    spec = DimensionSpec("drowsiness", LEVELS, ("eye_closure",))
    policy = RuleBasedPolicy([spec], {"eye_closure": 1.0}, attack_frames=2, release_frames=20)

    high = CueResult("eye_closure", "drowsiness", 1.0, True, "")
    calm = CueResult("eye_closure", "drowsiness", 0.0, False, "")

    policy.decide(_obs(), [calm])  # 初期値 0 を置く
    risen = policy.decide(_obs(), [high]).dimensions["drowsiness"].alarm
    fallen_from = risen
    fallen = policy.decide(_obs(), [calm]).dimensions["drowsiness"].alarm

    assert risen > 0.5  # 1フレームで半分以上まで立ち上がる
    assert fallen_from - fallen < risen  # 落ちる量は立ち上がりより小さい


def test_level_needs_margin_to_step_down():
    # 境界に張り付いた入力で段が往復すると、警告音が鳴り止み鳴り直す。
    spec = DimensionSpec("drowsiness", LEVELS, ("eye_closure",), release_margin=0.08)
    policy = _policy([spec], {"eye_closure": 1.0})

    def level_at(score):
        cue = CueResult("eye_closure", "drowsiness", score, True, "")
        return policy.decide(_obs(), [cue]).dimensions["drowsiness"].level

    assert level_at(0.62) == Level.MEDIUM
    assert level_at(0.58) == Level.MEDIUM  # 境界を少し割っても下げない
    assert level_at(0.50) == Level.LOW  # margin を超えて下がったら落ちる


def test_min_agree_discounts_a_lone_signal():
    # 3本の cue のうち1本だけが兆候を出しても、満額の警告にはしない。
    spec = DimensionSpec(
        "stress",
        LEVELS,
        ("hr_elevation", "facial_tension", "respiration"),
        combine="weighted",
        min_agree=2,
    )
    policy = _policy([spec], {"hr_elevation": 1.0, "facial_tension": 1.0, "respiration": 1.0})
    lone = policy.decide(
        _obs(),
        [
            CueResult("hr_elevation", "stress", 1.0, True, ""),
            CueResult("facial_tension", "stress", 0.0, False, ""),
            CueResult("respiration", "stress", 0.0, False, ""),
        ],
    )
    assert lone.dimensions["stress"].level < Level.MEDIUM


def test_invalid_cues_do_not_dilute_a_weighted_axis():
    # 測れていない cue を 0 として平均に数えると、残る cue がどれだけ強く出ても届かない。
    spec = DimensionSpec(
        "stress", LEVELS, ("hr_elevation", "facial_tension"), combine="weighted", min_agree=2
    )
    policy = _policy([spec], {"hr_elevation": 1.0, "facial_tension": 1.0})
    result = policy.decide(
        _obs(),
        [
            CueResult("hr_elevation", "stress", 0.0, False, "心拍なし", None, False),
            CueResult("facial_tension", "stress", 0.9, False, "", None, True),
        ],
    )
    # 有効なのは1本だけなので min_agree=2 に届かず割り引かれるが、
    # 無効な cue のぶんまで薄められて 0.45 未満に潰れることはない。
    assert result.dimensions["stress"].alarm == pytest.approx(0.45)


def test_reset_clears_smoothing_and_levels():
    spec = DimensionSpec("drowsiness", LEVELS, ("eye_closure",))
    policy = _policy([spec], {"eye_closure": 1.0})
    policy.decide(_obs(), [CueResult("eye_closure", "drowsiness", 1.0, True, "")])
    policy.reset()
    after = policy.decide(_obs(), [CueResult("eye_closure", "drowsiness", 0.0, False, "")])
    assert after.dimensions["drowsiness"].level == Level.NONE


STRESS_WEIGHTS = {"hr_elevation": 1.0, "facial_tension": 1.0, "respiration": 0.8, "blink_rate": 0.6}


def _stress_policy(min_agree: int = 2):
    spec = DimensionSpec(
        "stress", LEVELS, tuple(STRESS_WEIGHTS), combine="weighted", min_agree=min_agree
    )
    return _policy([spec], STRESS_WEIGHTS)


def _stress_alarm(policy, scores: dict[str, float]) -> float:
    cues = [CueResult(name, "stress", scores.get(name, 0.0), False, "") for name in STRESS_WEIGHTS]
    return policy.decide(_obs(), cues).dimensions["stress"].alarm


def test_min_agree_cues_give_the_full_warning_among_many_valid_cues():
    # 4 本とも計測できていて 2 本が満点。全部の平均だと 0.59 で LOW に留まり鳴らない。
    alarm = _stress_alarm(_stress_policy(), {"hr_elevation": 1.0, "facial_tension": 1.0})
    assert alarm == pytest.approx(1.0)


def test_min_agree_keeps_a_lone_cue_below_low_whatever_its_weight():
    policy = _stress_policy()
    for name in STRESS_WEIGHTS:
        assert _stress_alarm(policy, {name: 1.0}) < LEVELS["low"], name
    assert _stress_alarm(policy, {"hr_elevation": 1.0}) == pytest.approx(0.25)


def test_min_agree_zero_averages_every_valid_cue():
    scores = {"hr_elevation": 1.0, "facial_tension": 1.0}
    assert _stress_alarm(_stress_policy(min_agree=0), scores) == pytest.approx(2.0 / 3.4)


def _step_response(fps: float, seconds: float, **smoothing):
    """1 秒満点を入れてから 0 に落とし、(時刻, alarm, 段) の列を返す。"""
    spec = DimensionSpec("drowsiness", LEVELS, ("eye_closure",))
    policy = RuleBasedPolicy([spec], {"eye_closure": 1.0}, **smoothing)
    out = []
    for i in range(int(seconds * fps)):
        t = i / fps
        score = 1.0 if t < 1.0 else 0.0
        cue = CueResult("eye_closure", "drowsiness", score, score > 0, "")
        dim = policy.decide(make_observation(Features({}, t)), [cue]).dimensions["drowsiness"]
        out.append((t, dim.alarm, dim.level))
    return out


def _cleared_after(response) -> float:
    return next(t for t, _, level in response if t >= 1.0 and level == Level.NONE) - 1.0


def test_release_takes_the_same_time_at_any_frame_rate():
    # 1 フレームあたりの係数だと、同じ設定で 5fps の解除が 30fps の 6 倍かかる。
    fast = _cleared_after(_step_response(30.0, 4.0, attack_frames=2, release_frames=20))
    slow = _cleared_after(_step_response(5.0, 4.0, attack_frames=2, release_frames=20))
    assert 0.4 <= fast <= 0.7
    assert abs(slow - fast) <= 0.2 + 1e-9  # 5fps の 1 フレーム以内


def test_frame_settings_match_the_per_frame_ema_at_the_reference_rate():
    response = _step_response(30.0, 1.5, attack_frames=2, release_frames=20)
    expected = response[0][1]  # 最初の標本は平滑せずそのまま置く
    for t, alarm, _ in response[1:]:
        target = 1.0 if t < 1.0 else 0.0
        alpha = 2.0 / 3.0 if target > expected else 2.0 / 21.0
        expected += alpha * (target - expected)
        assert alarm == pytest.approx(expected, abs=1e-9)


def test_seconds_override_the_frame_settings():
    response = _step_response(10.0, 3.0, attack_seconds=0.0, release_seconds=1.0)
    at = {round(t, 1): alarm for t, alarm, _ in response}
    assert at[0.9] == pytest.approx(1.0)
    assert at[1.0] == pytest.approx(math.exp(-0.1))  # 0.1 秒で 1-exp(-0.1/1.0) だけ下がる
    assert at[2.0] == pytest.approx(math.exp(-1.1))


def test_a_timestamp_that_does_not_advance_counts_as_one_reference_frame():
    # 時刻が進まない、または戻った入力でも平滑値が止まらないこと。
    spec = DimensionSpec("drowsiness", LEVELS, ("eye_closure",))
    policy = RuleBasedPolicy([spec], {"eye_closure": 1.0}, attack_frames=2, release_frames=20)
    calm = CueResult("eye_closure", "drowsiness", 0.0, False, "")
    high = CueResult("eye_closure", "drowsiness", 1.0, True, "")
    policy.decide(make_observation(Features({}, 5.0)), [calm])
    same = policy.decide(make_observation(Features({}, 5.0)), [high])
    assert same.dimensions["drowsiness"].alarm == pytest.approx(2.0 / 3.0)
    back = policy.decide(make_observation(Features({}, 4.0)), [high])
    assert back.dimensions["drowsiness"].alarm == pytest.approx(1.0 - (1.0 / 3.0) ** 2)
