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


def _assessment(progress: float | None = 0.5) -> Assessment:
    dims = {"stress": Dimension("stress", 0.6, Level.MEDIUM, ("hr_elevation",))}
    cue = CueResult("hr_elevation", "stress", 0.6, True, "HR 82 base 68", progress)
    return Assessment(dimensions=dims, timestamp=0.0, cues=(cue,))


def _top_right_pixels(img: np.ndarray) -> int:
    region = img[40:160, img.shape[1] - 240 :]
    return int((region.sum(axis=2) > 0).sum())


def test_stress_meter_draws_progress_in_top_right():
    obs = _obs({"hr_bpm": 82.0, "rppg_quality": 0.34})
    a = _assessment(progress=0.5)
    with_meter = overlay.render(obs, a, stress_meter=True)
    without = overlay.render(obs, a, stress_meter=False)
    assert with_meter.shape == (480, 640, 3)
    # 進行度バーを出した方が右上に描画が増える。
    assert _top_right_pixels(with_meter) > _top_right_pixels(without)


def test_stress_meter_hidden_when_no_calibration():
    # progress を持たない cue（較正不要/固定基準）ならバーは出さない。
    obs = _obs({})
    a = _assessment(progress=None)
    assert _top_right_pixels(overlay.render(obs, a, stress_meter=True)) == 0


def test_stress_meter_full_when_calibrated():
    obs = _obs({"hr_bpm": 70.0, "rppg_quality": 0.4})
    a = _assessment(progress=1.0)
    img = overlay.render(obs, a, stress_meter=True)
    assert _top_right_pixels(img) > 0


def test_window_sink_scales_display_only():
    """表示の拡大は撮影解像度を変えない（rPPG は実効fpsで決まるため撮影は軽いまま）。"""
    import numpy as np

    from alertness.feedback.window_sink import OpenCvWindowSink

    sink = OpenCvWindowSink(window_width=1280)
    small = np.zeros((480, 640, 3), dtype=np.uint8)
    fitted = sink._fit(small)
    assert fitted.shape[1] == 1280
    assert fitted.shape[0] == 960  # 縦横比は保たれる
    assert small.shape == (480, 640, 3)  # 元の画像は変わらない

    assert OpenCvWindowSink(window_width=0)._fit(small).shape == small.shape
