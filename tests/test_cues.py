"""cue（特徴ごとの判定）のテスト。履歴を差し込んで時系列判定を確認する。"""

from __future__ import annotations

from _helpers import FakeHistory, make_observation

from alertness.classifier.cues.eye_closure import EyeClosureCue
from alertness.classifier.cues.gaze_off import GazeOffCue
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
