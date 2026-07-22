"""cue（特徴ごとの判定）のテスト。履歴を差し込んで時系列判定を確認する。"""

from __future__ import annotations

from _helpers import FakeHistory, make_observation

from alertness.classifier.cues.attention_hold import AttentionHoldCue
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


def test_attention_hold_active_when_gaze_and_head_steady():
    frames = [Features({"gaze_off": 0.01, "yaw_rel": 2.0}, i * 0.1) for i in range(50)]
    obs = make_observation(frames[-1], FakeHistory(frames))
    cue = AttentionHoldCue(gaze_on_threshold=0.035, steady_yaw_deg=12.0, sustained_seconds=3.0)
    result = cue.evaluate(obs)
    assert result.active
    assert result.score >= 1.0


def test_attention_hold_does_not_read_short_history_as_no_focus():
    # 起動直後（履歴 0.5 秒）でもずっと注視していれば低スコアにしない。集中は低いほど警告
    # する軸なので、履歴不足を「集中していない」と読むと必ず誤警告になる。
    frames = [Features({"gaze_off": 0.01, "yaw_rel": 2.0}, i * 0.1) for i in range(5)]
    obs = make_observation(frames[-1], FakeHistory(frames))
    cue = AttentionHoldCue(sustained_seconds=3.0)
    result = cue.evaluate(obs)
    assert result.score >= 1.0
    assert not result.active  # 3秒には届いていないので「集中と断定」はしない


def test_attention_hold_inactive_when_looking_away():
    frames = [Features({"gaze_off": 0.2, "yaw_rel": 30.0}, i * 0.1) for i in range(50)]
    obs = make_observation(frames[-1], FakeHistory(frames))
    assert not AttentionHoldCue().evaluate(obs).active


def test_hr_elevation_active_when_hr_elevated():
    # 固定基準（adaptive_baseline=false）なら較正不要で、すぐ baseline_bpm=70 と比べる。
    frames = [Features({"hr_bpm": 100.0, "rppg_quality": 0.5}, i * 0.1) for i in range(60)]
    obs = make_observation(frames[-1], FakeHistory(frames))
    cue = HrElevationCue(baseline_bpm=70.0, span_bpm=25.0, adaptive_baseline=False)
    result = cue.evaluate(obs)
    assert result.active
    assert result.score >= 1.0


def test_hr_elevation_silent_until_baseline_ready():
    # 安静基準が出来る前は、平常心拍が高い人でも「ストレス高」と出してはいけない。
    frames = [Features({"hr_bpm": 100.0, "rppg_quality": 0.5}, i * 0.1) for i in range(60)]
    obs = make_observation(frames[-1], FakeHistory(frames))
    result = HrElevationCue(baseline_bpm=70.0, span_bpm=25.0).evaluate(obs)
    assert not result.active
    assert result.score == 0.0
    assert "測定中" in result.detail


def test_hr_elevation_holds_value_while_head_moves():
    # 基準確立後に高ストレスを出し、その直後に頭が動くと 0 に落とさず値を保つ。
    steady = {"hr_bpm": 70.0, "rppg_quality": 0.5, "yaw": 0.0}
    calm = [Features(steady, i * 0.1) for i in range(500)]
    spike = [
        Features({"hr_bpm": 100.0, "rppg_quality": 0.5, "yaw": 0.0}, 50.0 + i * 0.1)
        for i in range(60)
    ]
    cue = HrElevationCue(span_bpm=25.0, baseline_seconds=45.0)
    frames = calm + spike
    assert cue.evaluate(make_observation(frames[-1], FakeHistory(frames))).score >= 1.0

    moving = [
        Features({"hr_bpm": 100.0, "rppg_quality": 0.5, "yaw": 30.0 * (-1) ** i}, 56.0 + i * 0.1)
        for i in range(30)
    ]
    frames = calm + spike + moving
    result = cue.evaluate(make_observation(frames[-1], FakeHistory(frames)))
    assert not result.active  # 保持中は根拠として数えない
    assert result.score >= 1.0  # それでも直前の値は保つ
    assert "頭部が動いている" in result.detail


def test_hr_elevation_adaptive_baseline_ignores_steady_high_hr():
    # 長時間ずっと同じ高さ（安静が高心拍の人）だと、本人基準に対して上がっていない＝none。
    frames = [Features({"hr_bpm": 95.0, "rppg_quality": 0.5}, i * 0.1) for i in range(600)]
    obs = make_observation(frames[-1], FakeHistory(frames))
    cue = HrElevationCue(span_bpm=25.0, baseline_seconds=45.0)
    assert not cue.evaluate(obs).active


def test_hr_elevation_adaptive_detects_rise_over_resting():
    # 安静(70)がしばらく続いた後に上昇(100)すると、本人基準からの上振れを拾う。
    calm = [Features({"hr_bpm": 70.0, "rppg_quality": 0.5}, i * 0.1) for i in range(500)]
    spike = [Features({"hr_bpm": 100.0, "rppg_quality": 0.5}, 50.0 + i * 0.1) for i in range(60)]
    frames = calm + spike
    obs = make_observation(frames[-1], FakeHistory(frames))
    cue = HrElevationCue(span_bpm=25.0, baseline_seconds=45.0)
    assert cue.evaluate(obs).active


def test_hr_elevation_inactive_without_rppg():
    # hr_bpm が無い（rPPG無効）と、ストレスを断定しない。
    frames = [Features({"ear": 0.3}, i * 0.1) for i in range(60)]
    obs = make_observation(frames[-1], FakeHistory(frames))
    result = HrElevationCue().evaluate(obs)
    assert not result.active
    assert result.score == 0.0


def test_hr_elevation_uses_hrv_when_available():
    # HRV(RMSSD)が本人基準(高RMSSD)より下がると、HRV経由でストレスを出す。
    calm = [Features({"hrv_rmssd": 60.0, "rppg_quality": 0.4}, i * 0.1) for i in range(500)]
    stressed = [
        Features({"hrv_rmssd": 25.0, "rppg_quality": 0.4}, 50.0 + i * 0.1) for i in range(60)
    ]
    frames = calm + stressed
    obs = make_observation(frames[-1], FakeHistory(frames))
    result = HrElevationCue(rmssd_span=25.0).evaluate(obs)
    assert result.detail.startswith("HRV")
    assert result.active


def test_hr_elevation_falls_back_to_hr_without_hrv():
    # HRV 標本が無ければ HR に退避する。
    frames = [Features({"hr_bpm": 100.0, "rppg_quality": 0.4}, i * 0.1) for i in range(60)]
    obs = make_observation(frames[-1], FakeHistory(frames))
    result = HrElevationCue(baseline_bpm=70.0, adaptive_baseline=False).evaluate(obs)
    assert result.detail.startswith("HR ")  # HRV ではなく HR 経由
    assert result.active
