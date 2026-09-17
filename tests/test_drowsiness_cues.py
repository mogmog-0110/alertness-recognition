"""早期の眠気兆候（瞬きの遅さ・うなずき・見失い）の cue のテスト。"""

from __future__ import annotations

from _helpers import FakeHistory, make_observation

from alertness.classifier.cues._episodes import closure_episodes
from alertness.classifier.cues.blink_dynamics import BlinkDynamicsCue
from alertness.classifier.cues.face_absent import FaceAbsentCue
from alertness.classifier.cues.nodding import NoddingCue
from alertness.contracts import Features

STEP = 0.05  # 20fps 相当。瞬きを刻める細かさ


def _blink_series(closed_seconds: float, count: int, gap_seconds: float = 3.0, tail: float = 0.5):
    """開眼→閉眼→開眼を count 回繰り返す ear_norm 列を作る。最後の瞬きの後は tail 秒だけ開ける。"""
    values: list[float] = []
    for _ in range(count):
        values += [1.0] * int(gap_seconds / STEP)
        values += [0.2] * max(1, int(closed_seconds / STEP))
    values += [1.0] * int(tail / STEP)
    return [Features({"ear_norm": v, "yaw_rel": 0.0}, i * STEP) for i, v in enumerate(values)]


def _evaluate(cue, frames):
    return cue.evaluate(make_observation(frames[-1], FakeHistory(frames)))


def test_closure_episodes_are_not_split_by_threshold_jitter():
    # 入口と出口が同じしきい値だと、境界をまたぐ震えで1回の閉眼が複数に割れる。
    times = [i * 0.05 for i in range(8)]
    ears = [1.0, 0.55, 0.65, 0.55, 0.3, 0.65, 1.0, 1.0]
    assert len(closure_episodes(times, ears, 0.6, 0.7)) == 1


def _episodes_of(ears: list[float]):
    times = [i * STEP for i in range(len(ears))]
    return closure_episodes(times, ears, 0.6, 0.7)


def test_reopen_time_never_reaches_past_the_next_blink():
    # 0.4 までしか戻らずにすぐ次の瞬きへ入る。先へ探し続けると、次の瞬きの向こう側の
    # 時刻が付き、瞬きの間隔が戻りの遅さとして数えられる。
    ears = [1.0] * 10 + [0.2] * 3 + [0.75] + [0.55] * 3 + [0.2] * 3 + [1.0] * 20
    first, second = _episodes_of(ears)
    assert first.reopened is not None and first.reopened < second.start
    assert second.reopen_seconds is not None


def test_reopen_time_ignores_the_slow_tail_of_the_landmarks():
    # ランドマークの値は、まぶたが開いた後も数百 ms かけて元の値へ漸近する。
    # 9 割のような高い目標で測ると、この尾の長さを「戻りが遅い」と読んでしまう。
    tail = [0.7 + 0.3 * (1 - 0.8**i) for i in range(30)]
    ears = [1.0] * 10 + [0.2] * 2 + [0.6] + tail
    first = _episodes_of(ears)[0]
    assert first.reopen_seconds is not None
    assert first.reopen_seconds <= 0.15


def test_reopen_time_keeps_a_slow_rise_long():
    # まぶたそのものの持ち上がりが遅い（底から 1 秒かけて開く）ときは長く出る。
    rise = [0.2 + 0.8 * i / 20 for i in range(1, 21)]
    ears = [1.0] * 10 + [0.2] * 3 + rise + [1.0] * 20
    first = _episodes_of(ears)[0]
    assert first.reopen_seconds is not None
    assert first.reopen_seconds >= 0.6


def test_reopen_target_follows_the_opening_before_the_blink():
    # 普段の開き具合が校正時より細い人（0.8）でも、瞬き直前の値を基準に同じ物差しで測る。
    ears = [0.8] * 10 + [0.2] * 2 + [0.52] + [0.8] * 20
    first = _episodes_of(ears)[0]
    assert first.reopen_seconds is not None
    assert first.reopen_seconds <= 0.15


def test_closure_times_fall_between_frames():
    # 端末経由の 15〜25fps では、フレームに丸めると 1 回の瞬きに数十 ms の段差が乗る。
    times = [0.0, 0.1, 0.2, 0.3, 0.4]
    ears = [1.0, 0.2, 0.2, 1.0, 1.0]
    (closure,) = closure_episodes(times, ears, 0.6, 0.7)
    assert abs(closure.start - 0.05) < 1e-9
    assert abs(closure.end - 0.2625) < 1e-9


def test_blink_dynamics_quiet_on_normal_blinks():
    frames = _blink_series(closed_seconds=0.12, count=5)
    result = _evaluate(BlinkDynamicsCue(), frames)
    assert result.score < 0.3
    assert not result.active


def test_blink_dynamics_fires_on_slow_blinks():
    # 覚醒時の3倍近い閉眼が続く＝眠気の早期兆候。
    frames = _blink_series(closed_seconds=0.40, count=5)
    result = _evaluate(BlinkDynamicsCue(), frames)
    assert result.score >= 0.85  # 最後の遅い瞬きから 0.5 秒ぶんだけ下がる
    assert result.active


def test_blink_dynamics_releases_after_the_slow_blinks_stop():
    # 遅い瞬きを続けた後に目を開けて普通に瞬きしても、窓の中央値は数十秒「遅い」のまま
    # 動かない。最後の遅い瞬きからの経過で下げ、数秒で警告を解く。
    slow = _blink_series(closed_seconds=0.40, count=8)
    normal = _blink_series(closed_seconds=0.10, count=3, tail=1.0)
    offset = slow[-1].timestamp + STEP
    frames = slow + [Features(f.values, f.timestamp + offset) for f in normal]
    result = _evaluate(BlinkDynamicsCue(), frames)
    assert not result.active
    assert result.score < 0.3


def test_blink_dynamics_waits_for_enough_blinks():
    # 1回の外れがそのまま判定にならないこと。
    frames = _blink_series(closed_seconds=0.40, count=1)
    result = _evaluate(BlinkDynamicsCue(min_blinks=3), frames)
    assert result.score == 0.0
    assert not result.valid


def test_blink_dynamics_ignores_sideways_face():
    frames = _blink_series(closed_seconds=0.40, count=5)
    turned = Features({"ear_norm": 1.0, "yaw_rel": 40.0}, frames[-1].timestamp)
    result = BlinkDynamicsCue().evaluate(make_observation(turned, FakeHistory([*frames, turned])))
    assert not result.valid


def _pitch_series(pattern: list[float]):
    return [Features({"pitch_rel": v}, i * 0.1) for i, v in enumerate(pattern)]


def test_nodding_counts_quick_drops_that_recover():
    # 12度落ちて 0.5 秒で戻る動きを3回。
    pattern: list[float] = []
    for _ in range(3):
        pattern += [0.0] * 20 + [12.0] * 5 + [0.0] * 20
    result = _evaluate(NoddingCue(nods_drowsy=3), _pitch_series(pattern))
    assert result.active
    assert "3回" in result.detail


def test_nodding_ignores_a_sustained_head_down():
    # 下を向いたまま戻らないのは居眠り姿勢。head_down の担当なのでここでは数えない。
    pattern = [0.0] * 20 + [12.0] * 200
    result = _evaluate(NoddingCue(max_seconds=2.5), _pitch_series(pattern))
    assert not result.active
    assert "0回" in result.detail


def test_nodding_quiet_when_head_is_still():
    result = _evaluate(NoddingCue(), _pitch_series([0.0] * 200))
    assert result.score == 0.0


def test_face_absent_fires_after_the_configured_delay():
    frames = [Features({}, i * 0.1, face_present=False) for i in range(60)]
    result = _evaluate(FaceAbsentCue(absent_seconds=3.0), frames)
    assert result.active
    assert result.score >= 1.0


def test_face_absent_is_quiet_while_the_face_is_visible():
    frames = [Features({"ear_norm": 1.0}, i * 0.1) for i in range(60)]
    result = _evaluate(FaceAbsentCue(), frames)
    assert result.score == 0.0
    assert not result.active


def test_face_absent_tolerates_a_single_dropped_frame():
    # 1フレームの検出漏れで警告を出してはいけない。
    frames = [Features({"ear_norm": 1.0}, i * 0.1) for i in range(60)]
    frames.append(Features({}, 6.0, face_present=False))
    result = _evaluate(FaceAbsentCue(grace_seconds=0.5), frames)
    assert not result.active
    assert result.score == 0.0


def test_nodding_fades_after_you_stop():
    """うなずくのをやめたら、窓の長さを待たずに下がる。

    箱型の窓だけだと、姿勢を直しても最大で窓の長さぶん警告が残る
    (実測: 眠気の警告が最長 30 秒続いた)。直したのに鳴り続ける警告は
    警告として働かない。
    """
    pattern: list[float] = []
    for _ in range(3):
        pattern += [0.0] * 20 + [12.0] * 5 + [0.0] * 20
    fresh = _evaluate(NoddingCue(nods_drowsy=3), _pitch_series(pattern))
    assert fresh.active

    # うなずいたあと、静止したまま 20 秒経過させる (1 サンプル 0.1 秒なので 200 個)。
    # 窓は 60 秒なので、うなずき自体はまだ窓の中に残っている。
    stale = _evaluate(NoddingCue(nods_drowsy=3), _pitch_series(pattern + [0.0] * 200))
    assert "3回" in stale.detail, "回数の数え方は変えない"
    assert not stale.active, "止めたら警告は下りる"
    assert stale.score < fresh.score / 2, "スコアも下がる"


def test_blink_dynamics_survives_one_long_closure():
    """1 回の長い閉眼で判定が跳ねない。

    窓に入る瞬きは数回しかないので、平均だと外れ値 1 個がそのまま判定になる
    (実測: 中央値 100ms のところ平均が 422ms まで上がり、覚醒しているのに
    眠気の警告が立ち続けた)。
    """
    frames = _blink_series(closed_seconds=0.10, count=5)
    normal = BlinkDynamicsCue().evaluate(
        make_observation(frames[-1], FakeHistory(frames))
    )
    assert normal.score < 0.5, "普通の瞬きでは立たない"

    # 同じ並びの途中に、1 回だけ 1 秒の閉眼を混ぜる。
    values: list[float] = []
    for i in range(5):
        values += [1.0] * int(3.0 / STEP)
        closed = 1.0 if i == 2 else 0.10
        values += [0.2] * max(1, int(closed / STEP))
    values += [1.0] * int(3.0 / STEP)
    with_outlier = [
        Features({"ear_norm": v, "yaw_rel": 0.0}, i * STEP) for i, v in enumerate(values)
    ]
    result = BlinkDynamicsCue().evaluate(
        make_observation(with_outlier[-1], FakeHistory(with_outlier))
    )
    assert result.score < 0.9, "外れ値 1 個で満点にはしない"


def test_blink_dynamics_thresholds_match_the_measured_awake_range():
    """覚醒時の普通の瞬きで立たないこと。

    文献値 (通常 150ms) はこの構成には低すぎた。実測 (acted_long / 1状態90秒)
    では覚醒時の閉眼中央値が 233ms あり、普通に瞬きしているだけでスコアが
    0.72 まで上がっていた。眠気時は中央値 1667ms で、7 倍の差がある。
    """
    from alertness.classifier.cues.blink_dynamics import BlinkDynamicsCue

    tuned = dict(
        normal_seconds=0.40, drowsy_seconds=1.20,
        normal_reopen=0.20, drowsy_reopen=0.45,
    )
    awake = _blink_series(closed_seconds=0.233, count=6)
    result = BlinkDynamicsCue(**tuned).evaluate(
        make_observation(awake[-1], FakeHistory(awake))
    )
    assert result.score < 0.3, "覚醒時の実測中央値では立たない"

    drowsy = _blink_series(closed_seconds=1.667, count=6)
    result = BlinkDynamicsCue(**tuned).evaluate(
        make_observation(drowsy[-1], FakeHistory(drowsy))
    )
    assert result.score > 0.7, "眠気時の実測中央値では立つ"


def _nod_frames(eyes_closed_during_drop: bool):
    frames = []
    t = 0.0
    for _ in range(3):
        for pitch in [0.0] * 20 + [12.0] * 5 + [0.0] * 20:
            closing = eyes_closed_during_drop and pitch > 0
            frames.append(Features({"pitch_rel": pitch, "eye_open": 0.3 if closing else 1.0}, t))
            t += 0.1
    return frames


def test_nodding_with_open_eyes_is_not_drowsiness_when_gated():
    # 相づちや話しながらの頷きは目が開いたまま起きる。
    result = _evaluate(NoddingCue(nods_drowsy=3, eyes_closed_ratio=0.7), _nod_frames(False))
    assert not result.active
    assert "0回" in result.detail


def test_nodding_with_closing_eyes_still_counts_when_gated():
    result = _evaluate(NoddingCue(nods_drowsy=3, eyes_closed_ratio=0.7), _nod_frames(True))
    assert result.active
    assert "3回" in result.detail


def test_eye_openness_needs_both_signals_to_close():
    from alertness.features.ear import eye_openness

    assert eye_openness(0.4, 0.1, 0.1) > 0.6, "EAR だけが閉じていても開いている"
    assert eye_openness(1.0, 0.9, 0.9) == 1.0, "瞬きスコアだけでも閉じない"
    assert eye_openness(0.3, 0.9, 0.9) < 0.6, "両方が閉じていれば閉じる"
    assert eye_openness(0.4, None, None) == 0.4, "瞬きスコアが無ければ EAR のまま"


def _head_down_frames(eye_open: float):
    return [Features({"pitch_rel": 20.0, "eye_open": eye_open}, i * 0.1) for i in range(20)]


def test_looking_down_with_open_eyes_is_not_drowsiness_when_gated():
    # 手元のスマホや資料を見ているだけ。前を見ていない状態は注意散漫の側が拾う。
    from alertness.classifier.cues.head_down import HeadDownCue

    cue = HeadDownCue(pitch_down_deg=12, eyes_closed_ratio=0.7)
    result = _evaluate(cue, _head_down_frames(eye_open=1.0))
    assert not result.active
    assert result.score == 0.0


def test_dozing_with_the_head_down_still_counts_when_gated():
    from alertness.classifier.cues.head_down import HeadDownCue

    cue = HeadDownCue(pitch_down_deg=12, eyes_closed_ratio=0.7)
    result = _evaluate(cue, _head_down_frames(eye_open=0.3))
    assert result.active


def test_eyes_closed_before_looking_down_do_not_make_it_dozing():
    # 目を閉じてから目を開けて下を向いた。窓全体の閉眼率では通ってしまうので、
    # 下を向いていた間に目も閉じていたかで見る。
    from alertness.classifier.cues.head_down import HeadDownCue

    # 0〜1.0 秒は目を閉じ、0.6 秒から下を向き、1.1 秒からは目を開けたまま下を向いている。
    # 窓の閉眼率は 55% あるが、下を向いていた間に目も閉じていたのは 3 分の 1 ほど。
    frames = [
        Features({"pitch_rel": 20.0 if i >= 6 else 0.0, "eye_open": 0.3 if i <= 10 else 1.0},
                 i * 0.1)
        for i in range(20)
    ]
    cue = HeadDownCue(pitch_down_deg=12, sustained_seconds=2.0, eyes_closed_ratio=0.7)
    assert not _evaluate(cue, frames).active


def test_blink_dynamics_does_not_count_a_long_closure_as_a_blink():
    # 3 秒閉じたのは瞬きではない（blink cue が受け持つ）。瞬きの中央値に混ぜると、
    # 目を開けた後も数秒「まばたきが遅い」が立ち続ける。
    values: list[float] = []
    for closed in (0.10, 0.10, 0.10, 3.0, 3.0):
        values += [1.0] * int(3.0 / STEP) + [0.2] * int(closed / STEP)
    values += [1.0] * int(1.0 / STEP)
    frames = [Features({"ear_norm": v, "yaw_rel": 0.0}, i * STEP) for i, v in enumerate(values)]
    result = _evaluate(BlinkDynamicsCue(normal_seconds=0.40, drowsy_seconds=1.20), frames)
    assert not result.active


def _lost_after(yaw: float, pitch: float, eye_open: float, lost_seconds: float):
    """2 秒かけて向きを変え、そのまま顔を見失う。"""
    ahead = {"yaw_rel": 0.0, "pitch_rel": 0.0, "eye_open": 1.0}
    frames = [Features(ahead, i * 0.1) for i in range(20)]
    frames += [
        Features({"yaw_rel": yaw, "pitch_rel": pitch, "eye_open": eye_open}, 2.0 + i * 0.1)
        for i in range(5)
    ]
    lost = int(lost_seconds / 0.1)
    frames += [Features({}, 2.5 + i * 0.1, face_present=False) for i in range(lost)]
    return frames


def test_a_face_lost_after_turning_away_stays_a_distraction():
    # 横を向きすぎると顔の検出が外れて向きが測れない。そこで黙ると、脇見が注意散漫や
    # 「顔が映っていません」として鳴る。
    from alertness.classifier.cues.head_turn import HeadTurnCue

    frames = _lost_after(yaw=30.0, pitch=0.0, eye_open=1.0, lost_seconds=4.0)
    turned = _evaluate(HeadTurnCue(yaw_side_deg=15, lost_hold_seconds=10), frames)
    assert turned.active
    absent = _evaluate(FaceAbsentCue(explain_yaw_deg=15, explain_pitch_deg=12), frames)
    assert not absent.active


def test_a_face_lost_after_looking_down_is_left_to_inattention():
    frames = _lost_after(yaw=0.0, pitch=25.0, eye_open=1.0, lost_seconds=4.0)
    absent = _evaluate(FaceAbsentCue(explain_yaw_deg=15, explain_pitch_deg=12), frames)
    assert not absent.active
    assert "下を向いて" in absent.detail


def test_slumping_with_closed_eyes_is_still_a_lost_driver():
    # 目を閉じたまま前に崩れて顔が外れるのは、居眠りで最も危ない形。読み替えない。
    # 居眠りでは目が先に閉じ、そのあと頭が落ちる。
    def frame(t: float, pitch: float, eye_open: float) -> Features:
        return Features({"yaw_rel": 0.0, "pitch_rel": pitch, "eye_open": eye_open}, t)

    frames = [frame(i * 0.1, 0.0, 1.0) for i in range(15)]
    frames += [frame(1.5 + i * 0.1, 0.0, 0.3) for i in range(10)]
    frames += [frame(2.5 + i * 0.1, 25.0, 0.3) for i in range(5)]
    frames += [Features({}, 3.0 + i * 0.1, face_present=False) for i in range(40)]
    assert _evaluate(FaceAbsentCue(explain_yaw_deg=15, explain_pitch_deg=12), frames).active


def test_a_long_disappearance_is_reported_even_after_turning():
    # 向きを変えた続きと読むのは hold の間だけ。戻ってこなければ見失いとして知らせる。
    frames = _lost_after(yaw=30.0, pitch=0.0, eye_open=1.0, lost_seconds=12.0)
    cue = FaceAbsentCue(explain_yaw_deg=15, explain_pitch_deg=12, explained_hold_seconds=10)
    assert _evaluate(cue, frames).active


def test_looking_down_lowers_the_lids_but_is_not_a_slump():
    # 下を向くとまぶたも下がる（実機で 0.68）。見失う直前の目で見ると、手元を見ただけで
    # 「目を閉じて崩れた」と読み、注意散漫ではなく「顔が映っていません」が出ていた。
    frames = _lost_after(yaw=0.0, pitch=18.0, eye_open=0.55, lost_seconds=4.0)
    absent = _evaluate(FaceAbsentCue(explain_yaw_deg=15, explain_pitch_deg=12), frames)
    assert not absent.active
