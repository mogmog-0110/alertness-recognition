"""小さい映像でも表示要素が重ならないことのテスト。

以前は座標をすべて画素数で直書きしていたため、映像が小さいと左上の判定パネルと
右下の debug 表示が重なり、文字が二重に見える不具合があった。
"""

from __future__ import annotations

import numpy as np
import pytest

from alertness.contracts import (
    Assessment,
    CalibrationProfile,
    CueResult,
    Dimension,
    FaceLandmarks,
    Features,
    Frame,
    Level,
    Observation,
)
from alertness.feedback import overlay
from alertness.feedback.timeline import DimensionTimeline

SIZES = [(1280, 720), (640, 480), (320, 240), (268, 160), (160, 120)]


def _observation(width: int, height: int) -> Observation:
    points = np.zeros((478, 3))
    points[33] = (0.42, 0.45, 0.0)
    points[263] = (0.58, 0.45, 0.0)
    landmarks = FaceLandmarks(points=points, image_size=(width, height), detected=True)
    values = {f"f{i}": 0.1 for i in range(29)}  # 実運用と同じくらいの特徴量の数
    features = Features(values, 0.0)

    class _History:
        fps = 30.0
        measured_fps = 30.0

        def latest(self):
            return features

        def recent(self, _seconds):
            return [features]

    frame = Frame(image=np.zeros((height, width, 3), dtype=np.uint8), index=0, timestamp=0.0)
    return Observation(frame, landmarks, features, _History(), CalibrationProfile.identity())


def _assessment() -> Assessment:
    dims = {
        "drowsiness": Dimension("drowsiness", 0.2, Level.NONE),
        "distraction": Dimension("distraction", 0.9, Level.HIGH, ("head_turn",)),
        "concentration": Dimension("concentration", 0.1, Level.HIGH, (), 0.9, "inattentive"),
        "stress": Dimension("stress", 0.3, Level.LOW),
    }
    cue = CueResult("hr_elevation", "stress", 0.3, False, "", 0.5, True)
    return Assessment(dimensions=dims, timestamp=0.0, cues=(cue,))


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_debug_panel_never_overlaps_the_judgement_panel(width, height):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert overlay.panel_right(image) + 8 <= width or width < 100


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_overlay_draws_at_any_size(width, height):
    obs = _observation(width, height)
    rendered = overlay.render(obs, _assessment(), True, True, False, True)
    assert rendered.shape == (height, width, 3)


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_timeline_and_calibration_fit_any_size(width, height):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    DimensionTimeline("stress", span_seconds=300.0).render(image, _assessment())
    assert overlay.draw_calibration(image, 0.5).shape == (height, width, 3)


def test_scale_shrinks_with_the_image():
    wide = np.zeros((720, 1280, 3), dtype=np.uint8)
    narrow = np.zeros((240, 320, 3), dtype=np.uint8)
    assert overlay.panel_right(narrow) < overlay.panel_right(wide)
