"""視線の水平比のテスト。

左右を見た動きに反応することが要件。目頭を基準に置くと左右の目で比が逆向きに出て、
平均が打ち消し合う（どちらを向いても 0.5 のまま）ので、そこを固定している。
"""

from __future__ import annotations

import numpy as np

from alertness.contracts import FaceLandmarks
from alertness.features import landmark_ids as ids
from alertness.features.gaze import horizontal_gaze_ratio

# 正面を向いた顔の目の並び（正規化座標）。画像の左に写るのが本人の右目。
_RIGHT_EYE = {ids.RIGHT_EYE_OUTER: 0.35, ids.RIGHT_EYE_INNER: 0.45, ids.RIGHT_IRIS: 0.40}
_LEFT_EYE = {ids.LEFT_EYE_INNER: 0.55, ids.LEFT_EYE_OUTER: 0.65, ids.LEFT_IRIS: 0.60}


def _face(drift: float = 0.0) -> FaceLandmarks:
    """虹彩を drift だけ画像の右へずらした顔。"""
    points = np.zeros((478, 3))
    points[:, 1] = 0.5
    for index, x in {**_RIGHT_EYE, **_LEFT_EYE}.items():
        points[index, 0] = x
    points[ids.LEFT_IRIS, 0] += drift
    points[ids.RIGHT_IRIS, 0] += drift
    return FaceLandmarks(points=points, image_size=(640, 480), detected=True)


def test_looking_straight_ahead_sits_in_the_middle():
    assert horizontal_gaze_ratio(_face()) == 0.5


def test_looking_right_and_left_move_in_opposite_directions():
    # 左右の目で比が打ち消し合うと、ここが両方 0.5 のままになる。
    assert horizontal_gaze_ratio(_face(0.012)) > 0.5
    assert horizontal_gaze_ratio(_face(-0.012)) < 0.5


def test_a_bigger_move_reads_bigger():
    small = horizontal_gaze_ratio(_face(0.006))
    large = horizontal_gaze_ratio(_face(0.012))
    assert 0.5 < small < large


def test_a_model_without_iris_points_says_nothing():
    points = np.zeros((468, 3))
    landmarks = FaceLandmarks(points=points, image_size=(640, 480), detected=True)
    assert horizontal_gaze_ratio(landmarks) == 0.5


def test_a_closed_up_eye_does_not_divide_by_zero():
    points = np.zeros((478, 3))
    assert horizontal_gaze_ratio(FaceLandmarks(points, (640, 480), True)) == 0.5
