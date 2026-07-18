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


def test_attention_hold_inactive_when_looking_away():
    frames = [Features({"gaze_off": 0.2, "yaw_rel": 30.0}, i * 0.1) for i in range(50)]
    obs = make_observation(frames[-1], FakeHistory(frames))
    assert not AttentionHoldCue().evaluate(obs).active


def test_hr_elevation_active_when_hr_elevated():
    frames = [Features({"hr_bpm": 100.0, "rppg_quality": 0.5}, i * 0.1) for i in range(60)]
    obs = make_observation(frames[-1], FakeHistory(frames))
    cue = HrElevationCue(baseline_bpm=70.0, span_bpm=30.0)
    result = cue.evaluate(obs)
    assert result.active
    assert result.score >= 1.0


def test_hr_elevation_inactive_without_rppg():
    # hr_bpm が無い（rPPG無効）と、ストレスを断定しない。
    frames = [Features({"ear": 0.3}, i * 0.1) for i in range(60)]
    obs = make_observation(frames[-1], FakeHistory(frames))
    result = HrElevationCue().evaluate(obs)
    assert not result.active
    assert result.score == 0.0
