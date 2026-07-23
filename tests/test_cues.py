"""cue（特徴ごとの判定）のテスト。履歴を差し込んで時系列判定を確認する。"""

from __future__ import annotations

import numpy as np
from _helpers import FakeHistory, make_observation

from alertness.classifier.cues.attention_buffer import AttentionBufferCue
from alertness.classifier.cues.eye_closure import EyeClosureCue
from alertness.classifier.cues.gaze_off import GazeOffCue
from alertness.classifier.cues.hr_elevation import HrElevationCue
from alertness.contracts import Features


def test_eye_closure_active_on_high_perclos():
    frames = [Features({"ear_norm": 0.2}, i * 0.1) for i in range(50)]
    obs = make_observation(frames[-1], FakeHistory(frames))
    cue = EyeClosureCue(window_seconds=30, perclos_drowsy=0.4, closed_ratio=0.6)
    result = cue.evaluate(obs)
    assert result.active
    assert result.score >= 1.0


def test_eye_closure_inactive_when_eyes_open():
    frames = [Features({"ear_norm": 1.0}, i * 0.1) for i in range(50)]
    obs = make_observation(frames[-1], FakeHistory(frames))
    cue = EyeClosureCue(window_seconds=30, perclos_drowsy=0.4, closed_ratio=0.6)
    assert not cue.evaluate(obs).active


def test_cue_inactive_without_face():
    obs = make_observation(Features({}, 0.0, face_present=False))
    result = EyeClosureCue().evaluate(obs)
    assert not result.active
    assert result.score == 0.0


def test_gaze_off_active_when_sustained():
    frames = [Features({"gaze_off": 0.4}, i * 0.1) for i in range(50)]
    obs = make_observation(frames[-1], FakeHistory(frames))
    cue = GazeOffCue(off_threshold=0.2, off_screen_seconds=2.0)
    assert cue.evaluate(obs).active


def _frames(gaze: float, yaw: float, n: int, t0: float = 0.0):
    return [Features({"gaze_off": gaze, "yaw_rel": yaw}, t0 + i * 0.1) for i in range(n)]


def test_attention_buffer_stays_full_while_on_target():
    cue = AttentionBufferCue(capacity_seconds=2.0)
    frames = _frames(0.01, 2.0, 50)
    for i in range(len(frames)):
        result = cue.evaluate(make_observation(frames[i], FakeHistory(frames[: i + 1])))
    assert result.score >= 1.0
    assert not result.active  # 残高があるので警告側には立たない


def test_attention_buffer_empties_when_looking_away():
    cue = AttentionBufferCue(capacity_seconds=2.0, latency_seconds=0.1)
    frames = _frames(0.30, 40.0, 40)  # 4秒よそ見（容量2秒＋猶予を超える）
    for i in range(len(frames)):
        result = cue.evaluate(make_observation(frames[i], FakeHistory(frames[: i + 1])))
    assert result.score == 0.0
    assert result.active


def test_attention_buffer_accumulates_across_repeated_glances():
    # 短いよそ見と復帰を繰り返す（視覚的時分割）。連続時間を数える方式では毎回満点に戻るが、
    # バッファ方式は戻りきらずに減っていく。ここが AttenD を選んだ理由。
    cue = AttentionBufferCue(capacity_seconds=2.0, latency_seconds=0.1, refill_rate=0.5)
    frames = []
    t = 0.0
    for _ in range(6):
        frames += _frames(0.30, 40.0, 8, t)  # 0.8秒よそ見
        t += 0.8
        frames += _frames(0.01, 2.0, 3, t)  # 0.3秒だけ戻る
        t += 0.3
    for i in range(len(frames)):
        result = cue.evaluate(make_observation(frames[i], FakeHistory(frames[: i + 1])))
    assert result.score < 0.5  # 直前に対象を見ていても残高は戻っていない


def test_attention_buffer_marks_invalid_without_face():
    cue = AttentionBufferCue()
    result = cue.evaluate(make_observation(Features({}, 0.0, face_present=False)))
    assert not result.valid


def _feed(cue, frames, step: int = 10):
    """フレームを順に食わせる。cue が安静基準を自前で育てるので、最終フレームだけでは足りない。"""
    result = None
    for i in range(0, len(frames), step):
        result = cue.evaluate(make_observation(frames[i], FakeHistory(frames[: i + 1])))
    return result


def _hr(bpm: float, seconds: float, t0: float = 0.0, quality: float = 0.7, **extra):
    n = int(seconds * 10)
    return [
        Features({"hr_bpm": bpm, "rppg_quality": quality, **extra}, t0 + i * 0.1) for i in range(n)
    ]


def test_hr_elevation_active_when_hr_elevated():
    # 固定基準（adaptive_baseline=false）なら較正不要で、すぐ baseline_bpm=70 と比べる。
    cue = HrElevationCue(baseline_bpm=70.0, span_bpm=25.0, adaptive_baseline=False)
    result = cue.evaluate(make_observation(_hr(100.0, 6.0)[-1], FakeHistory(_hr(100.0, 6.0))))
    assert result.active
    assert result.score >= 1.0


def test_hr_elevation_silent_until_baseline_ready():
    # 安静基準が出来る前は、平常心拍が高い人でも「ストレス高」と出してはいけない。
    frames = _hr(100.0, 6.0)
    result = HrElevationCue(baseline_bpm=70.0, span_bpm=25.0).evaluate(
        make_observation(frames[-1], FakeHistory(frames))
    )
    assert not result.active
    assert result.score == 0.0
    assert "測定中" in result.detail


def test_hr_elevation_motion_gate_uses_normalized_pose():
    # 生の pitch は ±180 を跨いで折り返す。そこを見ていると静止していても常に
    # 「動いている」判定になり、ストレスが永久に保持へ落ちる。
    frames = [
        Features(
            {
                "hr_bpm": 100.0,
                "rppg_quality": 0.7,
                "pitch": 180.0 - 360.0 * (i % 2),
                "pitch_rel": 0.0,
            },
            i * 0.1,
        )
        for i in range(1500)
    ]
    result = _feed(HrElevationCue(baseline_seconds=60.0), frames)
    assert "動いている" not in result.detail
    assert result.valid


def test_hr_elevation_ignores_low_quality_estimates():
    # 品質が閾値未満の推定は基準にも現在値にも使わない（誤差が span と同じ桁になるため）。
    frames = _hr(100.0, 200.0, quality=0.2)
    result = HrElevationCue().evaluate(make_observation(frames[-1], FakeHistory(frames)))
    assert not result.valid
    assert result.score == 0.0


def test_hr_elevation_stays_quiet_through_noisy_rest():
    # 心拍が変化していないのに推定だけが暴れている状態。ストレスを出してはいけない。
    rng = np.random.default_rng(0)
    frames = [
        Features({"hr_bpm": 72.0 + float(rng.normal(0, 15.0)), "rppg_quality": 0.7}, i * 0.1)
        for i in range(2000)
    ]
    cue = HrElevationCue(baseline_seconds=60.0)
    scores = []
    for i in range(0, len(frames), 10):
        result = cue.evaluate(make_observation(frames[i], FakeHistory(frames[: i + 1])))
        if frames[i].timestamp > 50.0:
            scores.append(result.score)
    # 誤警告率で見る。単発の外れではなく「鳴り続けないこと」が要件。
    assert np.mean(np.array(scores) >= 0.6) < 0.05


def test_hr_elevation_holds_baseline_through_long_elevation():
    # 上振れが基準の窓より長く続いても、基準が追いかけて見失わないこと。
    calm = _hr(70.0, 100.0)
    spike = _hr(100.0, 120.0, t0=100.0)  # 基準の窓(60秒)より長い上昇
    cue = HrElevationCue(span_bpm=25.0, baseline_seconds=60.0)
    result = _feed(cue, calm + spike)
    assert result.active
    assert result.score >= 0.8


def test_hr_elevation_holds_value_while_head_moves():
    # 基準確立後に高ストレスを出し、その直後に頭が動くと 0 に落とさず値を保つ。
    calm = _hr(70.0, 100.0, yaw_rel=0.0)
    spike = _hr(100.0, 10.0, t0=100.0, yaw_rel=0.0)
    cue = HrElevationCue(span_bpm=25.0, baseline_seconds=60.0)
    assert _feed(cue, calm + spike).score >= 0.8

    moving = [
        Features(
            {"hr_bpm": 100.0, "rppg_quality": 0.7, "yaw_rel": 30.0 * (-1) ** i}, 110.0 + i * 0.1
        )
        for i in range(30)
    ]
    result = _feed(cue, calm + spike + moving, step=1)
    assert not result.active  # 保持中は根拠として数えない
    assert result.score >= 0.8  # それでも直前の値は保つ
    assert "頭部が動いている" in result.detail


def test_hr_elevation_adaptive_baseline_ignores_steady_high_hr():
    # 長時間ずっと同じ高さ（安静が高心拍の人）だと、本人基準に対して上がっていない＝none。
    cue = HrElevationCue(span_bpm=25.0, baseline_seconds=60.0)
    assert not _feed(cue, _hr(95.0, 120.0)).active


def test_hr_elevation_adaptive_detects_rise_over_resting():
    # 安静(70)がしばらく続いた後に上昇(100)すると、本人基準からの上振れを拾う。
    cue = HrElevationCue(span_bpm=25.0, baseline_seconds=60.0)
    assert _feed(cue, _hr(70.0, 100.0) + _hr(100.0, 10.0, t0=100.0)).active


def test_hr_elevation_inactive_without_rppg():
    # hr_bpm が無い（rPPG無効）と、ストレスを断定しない。
    frames = [Features({"ear": 0.3}, i * 0.1) for i in range(60)]
    result = HrElevationCue().evaluate(make_observation(frames[-1], FakeHistory(frames)))
    assert not result.active
    assert result.score == 0.0


def test_hr_elevation_uses_hrv_when_available():
    # HRV(RMSSD)が本人基準より下がると、HRV経由でストレスを出す。
    calm = [Features({"hrv_rmssd": 60.0, "rppg_quality": 0.7}, i * 0.1) for i in range(500)]
    stressed = [
        Features({"hrv_rmssd": 20.0, "rppg_quality": 0.7}, 50.0 + i * 0.1) for i in range(60)
    ]
    frames = calm + stressed
    result = HrElevationCue(rmssd_span=25.0).evaluate(
        make_observation(frames[-1], FakeHistory(frames))
    )
    assert result.detail.startswith("HRV")
    assert result.active


def test_hr_elevation_falls_back_to_hr_without_hrv():
    # HRV 標本が無ければ HR に退避する。
    frames = _hr(100.0, 6.0)
    result = HrElevationCue(baseline_bpm=70.0, adaptive_baseline=False).evaluate(
        make_observation(frames[-1], FakeHistory(frames))
    )
    assert result.detail.startswith("HR ")  # HRV ではなく HR 経由
    assert result.active


def test_hr_elevation_baseline_survives_a_long_freeze():
    # 上振れで基準の更新を止めている間に時間が経っても、基準が消えないこと。
    # 消えると再開直後の数点で新しい基準ができ、そこに測り損ねた値が混ざると崩壊する。
    cue = HrElevationCue(span_bpm=10.0, baseline_seconds=60.0)
    _feed(cue, _hr(62.0, 100.0))
    base_before, _, ready = cue._read_baseline()
    assert ready

    # 基準の窓(60秒)より長く上振れが続く
    _feed(cue, _hr(62.0, 100.0) + _hr(75.0, 120.0, t0=100.0))
    base_after, _, ready = cue._read_baseline()
    assert ready
    assert abs(base_after - base_before) < 3.0  # 基準は安静のまま


def test_hr_elevation_rest_samples_are_time_spaced():
    # 毎フレーム積むと直近の1点が重複して基準を乗っ取る。間隔を空けて積むこと。
    cue = HrElevationCue(baseline_seconds=60.0, rest_interval=1.0)
    _feed(cue, _hr(62.0, 60.0), step=1)  # 0.1秒刻みで600フレーム
    assert len(cue._rest._samples) <= 65  # 1秒間隔なら60件前後。重複していれば600件になる
