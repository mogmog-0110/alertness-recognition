from __future__ import annotations

import numpy as np

from alertness.contracts import FaceLandmarks, Features, Frame, Observation
from alertness.feedback.rppg_view import RppgView


def _landmarks(size: int, detected: bool = True) -> FaceLandmarks:
    pts = np.zeros((470, 3))
    pts[33] = (0.25, 0.6, 0.0)
    pts[263] = (0.75, 0.6, 0.0)
    return FaceLandmarks(points=pts, image_size=(size, size), detected=detected)


def _obs(frame_img: np.ndarray, features: Features, lm: FaceLandmarks) -> Observation:
    frame = Frame(image=frame_img, index=0, timestamp=features.timestamp)

    class _H:
        fps = 30.0

        def latest(self):
            return features

        def recent(self, seconds):
            return [features]

    return Observation(frame=frame, landmarks=lm, features=features, history=_H(), profile=None)  # type: ignore[arg-type]


def _bottom_left(img: np.ndarray) -> int:
    h = img.shape[0]
    return int((img[h - 130 : h, 0:280].sum(axis=2) > 0).sum())


def test_rppg_view_draws_crop_and_waveform():
    view = RppgView(fps=30.0)
    size = 200
    lm = _landmarks(size)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(40):
        frame_img = np.zeros((size, size, 3), dtype=np.uint8)
        frame_img[:, :] = (100, int(128 + 20 * np.sin(i * 0.6)), 150)  # 脈動色
        obs = _obs(frame_img, Features({"hr_bpm": 72.0, "rppg_quality": 0.3}, i / 30.0), lm)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        view.render(canvas, obs)

    assert _bottom_left(canvas) > 0
    # 額切り出しのシアン枠 BGR(0,255,255) が描かれている。
    cyan = (canvas[:, :, 0] < 80) & (canvas[:, :, 1] > 200) & (canvas[:, :, 2] > 200)
    assert cyan.sum() > 0


def test_rppg_view_handles_no_face():
    view = RppgView()
    size = 200
    lm = _landmarks(size, detected=False)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    obs = _obs(np.zeros((size, size, 3), dtype=np.uint8), Features({}, 0.0), lm)
    view.render(canvas, obs)  # 顔なしでもクラッシュしない
    assert _bottom_left(canvas) > 0  # 少なくともタイトルは描かれる
