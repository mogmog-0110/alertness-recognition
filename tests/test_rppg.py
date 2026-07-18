from __future__ import annotations

import math

import numpy as np

from alertness.contracts import FaceLandmarks, Features, Frame
from alertness.features.rppg import RppgEstimator, estimate_hr, pos_signal


def _synthetic_rgb(hr_bpm: float, fs: float, seconds: float) -> np.ndarray:
    # 緑が最も強く脈打つ、DC 付きの RGB 時系列。実際の肌色に近い前提。
    t = np.arange(0, seconds, 1.0 / fs)
    pulse = np.sin(2.0 * np.pi * (hr_bpm / 60.0) * t)
    r = 0.6 + 0.005 * pulse
    g = 0.5 + 0.02 * pulse
    b = 0.4 + 0.003 * pulse
    return np.stack([r, g, b], axis=1)


def test_estimate_hr_recovers_synthetic_rate():
    fs = 30.0
    rgb = _synthetic_rgb(72.0, fs, 12.0)
    hr, quality = estimate_hr(pos_signal(rgb), fs)
    assert abs(hr - 72.0) < 4.0
    assert quality > 0.1


def test_estimate_hr_nan_when_too_short():
    hr, quality = estimate_hr(np.zeros(4), 30.0)
    assert math.isnan(hr)
    assert quality == 0.0


def test_pos_signal_is_zero_mean():
    rgb = _synthetic_rgb(66.0, 30.0, 8.0)
    sig = pos_signal(rgb)
    assert sig.shape[0] == rgb.shape[0]
    assert abs(float(np.mean(sig))) < 1e-9


def _landmarks_with_eyes(size: int) -> FaceLandmarks:
    # 目尻を左右に置き、額 ROI が画像中央上に来るようにする。
    points = np.zeros((470, 3))
    points[33] = (0.25, 0.6, 0.0)  # LEFT_EYE_OUTER
    points[263] = (0.75, 0.6, 0.0)  # RIGHT_EYE_OUTER
    return FaceLandmarks(points=points, image_size=(size, size), detected=True)


def test_augment_adds_hr_after_enough_frames():
    fs, size = 30.0, 64
    est = RppgEstimator(fps=fs, window_seconds=6.0)
    landmarks = _landmarks_with_eyes(size)
    rgb = _synthetic_rgb(72.0, fs, 6.0)

    out = None
    for i, (r, g, b) in enumerate(rgb):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        # 額 ROI 相当（画像中央上）に脈打つ色を置く。OpenCV は BGR。
        image[10:26, 20:44] = (int(b * 255), int(g * 255), int(r * 255))
        frame = Frame(image=image, index=i, timestamp=i / fs)
        out = est.augment(frame, landmarks, Features({}, i / fs))

    assert out is not None
    assert "hr_bpm" in out.values
    assert "rppg_quality" in out.values


def test_augment_noop_when_face_absent():
    est = RppgEstimator()
    landmarks = FaceLandmarks(points=np.zeros((470, 3)), image_size=(64, 64), detected=False)
    frame = Frame(image=np.zeros((64, 64, 3), dtype=np.uint8), index=0, timestamp=0.0)
    features = Features({"ear": 0.3}, 0.0)
    out = est.augment(frame, landmarks, features)
    assert out is features  # 検出なしなら素通し
