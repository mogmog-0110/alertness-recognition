"""表情の緊張（AU4/AU7/AU24）によるストレス手がかりのテスト。"""

from __future__ import annotations

from _helpers import FakeHistory, make_observation

from alertness.classifier.cues.facial_tension import FacialTensionCue
from alertness.contracts import Features

_AUS = ("browDownLeft", "browDownRight", "eyeSquintLeft", "eyeSquintRight")


def _frames(level: float, seconds: float, t0: float = 0.0):
    values = dict.fromkeys(_AUS, level)
    return [Features(values, t0 + i * 0.1) for i in range(int(seconds * 10))]


def _feed(cue, frames, step: int = 5):
    result = None
    for i in range(0, len(frames), step):
        result = cue.evaluate(make_observation(frames[i], FakeHistory(frames[: i + 1])))
    return result


def test_facial_tension_silent_until_baseline_ready():
    # 基準が出来る前は、もともと眉が寄っている人でも緊張と断定しない。
    result = _feed(FacialTensionCue(), _frames(0.5, 20.0))
    assert result.score == 0.0
    assert not result.valid
    assert "測定中" in result.detail


def test_facial_tension_quiet_at_own_baseline():
    # 基準どおりの表情が続いている間は 0。個人差は基準で吸収される。
    result = _feed(FacialTensionCue(baseline_seconds=60.0), _frames(0.5, 120.0))
    assert result.valid
    assert result.score < 0.1


def test_facial_tension_rises_above_own_baseline():
    # 本人の安静より眉が寄れば上がる。絶対値ではなく差で見ている。
    cue = FacialTensionCue(span=0.15, baseline_seconds=60.0)
    calm = _frames(0.10, 120.0)
    tense = _frames(0.35, 20.0, t0=120.0)
    result = _feed(cue, calm + tense)
    assert result.score >= 0.8


def test_facial_tension_never_asserts_alone():
    # 単独ではストレスを断定しない（active を立てない）。心拍と揃って初めて効かせる。
    cue = FacialTensionCue(span=0.15, baseline_seconds=60.0)
    result = _feed(cue, _frames(0.10, 120.0) + _frames(0.9, 20.0, t0=120.0))
    assert result.score >= 1.0
    assert not result.active


def test_facial_tension_handles_missing_blendshapes():
    # blendshape が無い環境（モデル未対応など）でも落ちず、断定もしない。
    frames = [Features({"ear": 0.3}, i * 0.1) for i in range(50)]
    result = _feed(FacialTensionCue(), frames)
    assert result.score == 0.0
    assert not result.valid


def test_facial_tension_resets_when_face_lost():
    cue = FacialTensionCue(baseline_seconds=60.0)
    _feed(cue, _frames(0.2, 120.0))
    assert cue.evaluate(make_observation(Features({}, 200.0, face_present=False))).valid is False
