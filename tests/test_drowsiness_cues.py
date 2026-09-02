"""早期の眠気兆候（瞬きの遅さ・うなずき・見失い）の cue のテスト。"""

from __future__ import annotations

from _helpers import FakeHistory, make_observation

from alertness.classifier.cues._episodes import closure_episodes
from alertness.classifier.cues.blink_dynamics import BlinkDynamicsCue
from alertness.classifier.cues.face_absent import FaceAbsentCue
from alertness.classifier.cues.nodding import NoddingCue
from alertness.contracts import Features

STEP = 0.05  # 20fps 相当。瞬きを刻める細かさ


def _blink_series(closed_seconds: float, count: int, gap_seconds: float = 3.0):
    """開眼→閉眼→開眼を count 回繰り返す ear_norm 列を作る。"""
    values: list[float] = []
    for _ in range(count):
        values += [1.0] * int(gap_seconds / STEP)
        values += [0.2] * max(1, int(closed_seconds / STEP))
    values += [1.0] * int(gap_seconds / STEP)
    return [Features({"ear_norm": v, "yaw_rel": 0.0}, i * STEP) for i, v in enumerate(values)]


def _evaluate(cue, frames):
    return cue.evaluate(make_observation(frames[-1], FakeHistory(frames)))


def test_closure_episodes_are_not_split_by_threshold_jitter():
    # 入口と出口が同じしきい値だと、境界をまたぐ震えで1回の閉眼が複数に割れる。
    times = [i * 0.05 for i in range(8)]
    ears = [1.0, 0.55, 0.65, 0.55, 0.3, 0.65, 1.0, 1.0]
    assert len(closure_episodes(times, ears, 0.6, 0.7)) == 1


def test_blink_dynamics_quiet_on_normal_blinks():
    frames = _blink_series(closed_seconds=0.12, count=5)
    result = _evaluate(BlinkDynamicsCue(), frames)
    assert result.score < 0.3
    assert not result.active


def test_blink_dynamics_fires_on_slow_blinks():
    # 覚醒時の3倍近い閉眼が続く＝眠気の早期兆候。
    frames = _blink_series(closed_seconds=0.40, count=5)
    result = _evaluate(BlinkDynamicsCue(), frames)
    assert result.score >= 1.0
    assert result.active


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
