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


def test_display_sizes_window_once_and_keeps_aspect(monkeypatch):
    """表示・キャリブ・ガイドの3経路が同じ窓設定を通ること（以前はガイドだけ効かなかった）。"""
    import numpy as np

    from alertness.feedback import display

    calls: list[tuple] = []
    monkeypatch.setattr(display.cv2, "namedWindow", lambda *a: calls.append(("named", *a)))
    monkeypatch.setattr(display.cv2, "resizeWindow", lambda *a: calls.append(("resize", *a)))
    monkeypatch.setattr(display.cv2, "imshow", lambda *a: calls.append(("imshow",)))

    display.reset()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    display.show(image, 1280)
    display.show(image, 1280)  # 2回目は大きさを触らない（手で変えた分を戻さない）

    sized = [c for c in calls if c[0] == "resize"]
    assert sized == [("resize", display.WINDOW_NAME, 1280, 960)]  # 縦横比は保たれる
    assert sum(1 for c in calls if c[0] == "imshow") == 2

    calls.clear()
    display.reset()
    display.show(image, 0)  # 0 なら撮影したままの大きさ
    assert not [c for c in calls if c[0] in ("named", "resize")]
