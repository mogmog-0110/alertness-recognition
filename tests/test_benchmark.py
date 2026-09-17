"""誤警告と見逃しの計測台のテスト。

計測台が壊れていると、しきい値の変更が改善か改悪か分からないまま入る。合成の
「起きている人」の収録で、誤警告を数えられることと、差し込んだ閉眼を検出できることを見る。
"""

from __future__ import annotations

import csv

import pytest

from alertness.config import load_config
from alertness.evaluation import benchmark
from alertness.evaluation.scenarios import SCENARIOS

_FPS = 10.0
_SECONDS = 110.0


def _awake_csv(path) -> str:
    """3 秒ごとに普通の瞬きをして、正面を見ている 110 秒。"""
    fields = ["timestamp", "face_present", "ear_norm", "eyeBlinkLeft", "eyeBlinkRight",
              "pitch_rel", "yaw_rel", "gaze_x", "gaze_off", "jawOpen"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(int(_SECONDS * _FPS)):
            t = i / _FPS
            blinking = (t % 3.0) < 0.15
            dx = 0.01 if (t % 7.0) < 3.5 else -0.01
            writer.writerow({
                "timestamp": f"{100.0 + t:.4f}", "face_present": "1",
                "ear_norm": "0.2" if blinking else "1.0",
                "eyeBlinkLeft": "0.9" if blinking else "0.05",
                "eyeBlinkRight": "0.9" if blinking else "0.05",
                "pitch_rel": "0.0", "yaw_rel": "0.0",
                "gaze_x": f"{0.5 + dx:.4f}", "gaze_off": f"{abs(dx):.4f}", "jawOpen": "0.0",
            })
    return str(path)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    path = _awake_csv(tmp_path_factory.mktemp("bench") / "awake.csv")
    config = load_config("config/browser.yaml")
    frames = benchmark.load_features(path)
    control = list(benchmark.replay(frames, config))
    return frames, config, control


def test_old_recordings_get_the_missing_columns(setup):
    frames, _, _ = setup
    # gaze_off=|gaze_x-中心| から中心 0.5 を逆算し、向き付きの gaze_dx を補う。
    assert frames[0].values["gaze_dx"] == pytest.approx(0.01)
    assert "eye_open" in frames[0].values


def test_a_calm_recording_raises_no_alarm(setup):
    _, config, control = setup
    report = benchmark.false_alarms(control, config)
    assert report["seconds"] > 40
    assert report["alert_share"].get("drowsiness", 0.0) == 0.0
    assert report["alert_share"].get("distraction", 0.0) == 0.0


def test_an_injected_microsleep_is_detected_quickly(setup):
    frames, config, control = setup
    microsleep = next(s for s in SCENARIOS if s.name == "microsleep")
    result = benchmark.detection(frames, config, microsleep, control)
    assert result["control"]["latency"] is None, "差し込まなければ上がらない"
    assert result["injected"]["latency"] is not None
    assert result["injected"]["latency"] <= 1.5


def test_every_scenario_names_a_configured_dimension():
    config = load_config("config/browser.yaml")
    names = {d["name"] for d in config["assessment"]["dimensions"]}
    assert {s.dimension for s in SCENARIOS} <= names


def test_the_gaze_center_tie_prefers_the_front():
    from alertness.evaluation.benchmark import _gaze_center

    rows = [{"gaze_x": "0.46", "gaze_off": "0.04"}] * 10  # 候補 0.42 と 0.50 が同票
    assert _gaze_center(rows) == pytest.approx(0.50)


def test_bad_arguments_are_rejected_before_starting(tmp_path, capsys):
    from alertness.benchmark import main

    with pytest.raises(SystemExit):
        main([str(tmp_path), "--jobs", "0"])
    assert main([str(tmp_path / "missing.csv")]) == 1
