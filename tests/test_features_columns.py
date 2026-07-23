"""特徴量の抽出・正規化・CSV列の対応。

映像を捨てたあとで取り込みをやり直さずに済ませるため、検出器が計算した値は選別せず
すべてCSVの列になっている必要がある。ここではその対応が崩れていないことを見る。
"""

from __future__ import annotations

import numpy as np
import pytest

from alertness.contracts import CalibrationProfile, FaceLandmarks, Features, Pose
from alertness.features.extractor import BLENDSHAPE_COLUMNS, FaceFeatureExtractor
from alertness.features.gaze import horizontal_gaze_ratio, vertical_gaze_ratio
from alertness.features.normalize import normalize_features
from alertness.feedback.csv_sink import FEATURE_COLUMNS


def _landmarks(iris_y: float = 0.50, iris_x: float = 0.50, blendshapes=None) -> FaceLandmarks:
    """478点のダミー。目頭・目尻・虹彩だけ意味のある位置に置く。"""
    points = np.full((478, 3), 0.5, dtype=float)
    for inner, outer in ((133, 33), (362, 263)):
        points[inner] = (0.40, 0.50, 0.0)  # 目頭
        points[outer] = (0.60, 0.50, 0.0)  # 目尻（同じ高さ＝基準線は水平）
    points[468] = (iris_x, iris_y, 0.0)  # 左虹彩
    points[473] = (iris_x, iris_y, 0.0)  # 右虹彩
    return FaceLandmarks(
        points=points,
        image_size=(1000, 1000),
        detected=True,
        blendshapes=blendshapes or {},
    )


def test_vertical_gaze_is_centred_on_the_eye_corner_line():
    # 虹彩が目頭・目尻を結ぶ線上にあるとき、上下どちらでもない。
    assert vertical_gaze_ratio(_landmarks(iris_y=0.50)) == pytest.approx(0.5)


def test_vertical_gaze_increases_when_looking_down():
    up = vertical_gaze_ratio(_landmarks(iris_y=0.48))
    centre = vertical_gaze_ratio(_landmarks(iris_y=0.50))
    down = vertical_gaze_ratio(_landmarks(iris_y=0.52))
    assert up < centre < down


def test_vertical_gaze_is_scaled_by_eye_width():
    # 目の幅で割るので、顔の遠近（画素サイズ）が変わっても同じ値になる。
    near = _landmarks(iris_y=0.52)
    far = FaceLandmarks(
        points=near.points, image_size=(500, 500), detected=True, blendshapes={}
    )
    assert vertical_gaze_ratio(near) == pytest.approx(vertical_gaze_ratio(far))


def test_vertical_gaze_defaults_without_iris_points():
    # 虹彩点を持たないモデルでは中央を返す（水平比と同じ扱い）。
    lm = FaceLandmarks(points=np.zeros((468, 3)), image_size=(100, 100), detected=True)
    assert vertical_gaze_ratio(lm) == 0.5
    assert horizontal_gaze_ratio(lm) == 0.5


def test_extractor_passes_every_blendshape_through():
    # 検出器が出した blendshape は選別せず全部通す。
    values = dict.fromkeys(BLENDSHAPE_COLUMNS, 0.25)
    features = FaceFeatureExtractor().extract(_landmarks(blendshapes=values), 0.0)

    assert len(BLENDSHAPE_COLUMNS) == 52
    for name in BLENDSHAPE_COLUMNS:
        assert features.get(name) == pytest.approx(0.25), name


def test_extractor_reports_head_position_and_vertical_gaze():
    features = FaceFeatureExtractor().extract(_landmarks(iris_y=0.52), 0.0)

    for name in ("gaze_x", "gaze_y", "head_x", "head_y", "face_scale"):
        assert not np.isnan(features.get(name)), name
    # 頭の位置は正規化座標なので 0..1。
    assert 0.0 <= features.get("head_x") <= 1.0
    assert 0.0 <= features.get("head_y") <= 1.0


def test_every_extracted_value_has_a_csv_column():
    values = dict.fromkeys(BLENDSHAPE_COLUMNS, 0.1)
    raw = FaceFeatureExtractor().extract(_landmarks(blendshapes=values), 0.0)
    profile = CalibrationProfile(
        ear_open_baseline=0.3,
        mar_neutral=0.0,
        head_pose_neutral=Pose(0.0, 0.0, 0.0),
        gaze_center=(0.5, 0.5),
        face_scale=1.0,
    )
    normalized = normalize_features(raw, profile)

    # CSV は extrasaction="ignore" なので、列が無い値は黙って捨てられる。
    missing = sorted(set(normalized.values) - set(FEATURE_COLUMNS))
    assert missing == [], f"CSVに列が無い特徴量: {missing}"


def test_vertical_gaze_offset_uses_the_calibrated_centre():
    raw = Features(values={"gaze_y": 0.7}, timestamp=0.0)
    profile = CalibrationProfile(
        ear_open_baseline=0.3,
        mar_neutral=0.0,
        head_pose_neutral=Pose(0.0, 0.0, 0.0),
        gaze_center=(0.5, 0.6),
        face_scale=1.0,
    )
    assert normalize_features(raw, profile).get("gaze_off_y") == pytest.approx(0.1)
