from __future__ import annotations

import numpy as np

from alertness.contracts import (
    Assessment,
    CueResult,
    Dimension,
    FaceLandmarks,
    Features,
    Frame,
    Level,
    Observation,
)
from alertness.feedback import overlay


def _obs(values: dict) -> Observation:
    frame = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), index=0, timestamp=0.0)
    lm = FaceLandmarks(points=np.zeros((0, 3)), image_size=(640, 480), detected=False)
    feats = Features(values, 0.0)

    class _H:
        fps = 30.0

        def latest(self):
            return feats

        def recent(self, seconds):
            return [feats]

    return Observation(frame=frame, landmarks=lm, features=feats, history=_H(), profile=None)  # type: ignore[arg-type]


def _assessment() -> Assessment:
    dims = {"stress": Dimension("stress", 0.6, Level.MEDIUM, ("hr_elevation",))}
    cue = CueResult("hr_elevation", "stress", 0.6, True, "HR 82 base 68")
    return Assessment(dimensions=dims, timestamp=0.0, cues=(cue,))


def _top_right_pixels(img: np.ndarray) -> int:
    region = img[40:160, img.shape[1] - 240 :]
    return int((region.sum(axis=2) > 0).sum())


def test_stress_meter_draws_in_top_right():
    obs = _obs({"hr_bpm": 82.0, "rppg_quality": 0.34})
    a = _assessment()
    with_meter = overlay.render(obs, a, stress_meter=True)
    without = overlay.render(obs, a, stress_meter=False)
    assert with_meter.shape == (480, 640, 3)
    # メーターを出した方が右上に描画が増える。
    assert _top_right_pixels(with_meter) > _top_right_pixels(without)


def test_stress_meter_shows_hint_when_rppg_off():
    # hr_bpm が無い（rPPG無効）でもクラッシュせず、メーター枠は描かれる。
    obs = _obs({})
    img = overlay.render(obs, _assessment(), stress_meter=True)
    assert _top_right_pixels(img) > 0
