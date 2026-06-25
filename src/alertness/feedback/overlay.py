"""映像へのオーバーレイ描画。

画面上の文字は OpenCV の制約で日本語を描けないため英字表記にしている
（コメントやログは日本語）。どんな背景でも読めるよう黒縁付きで描く。
debug=True のときは生の特徴量も出し、しきい値調整の手がかりにする。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..contracts import Assessment, FaceLandmarks, Level, Observation
from ..features import landmark_ids as ids

WINDOW_NAME = "Alertness"
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# レベルごとの色（BGR）
_COLORS = {
    Level.NONE: (0, 180, 0),
    Level.LOW: (0, 200, 200),
    Level.MEDIUM: (0, 140, 255),
    Level.HIGH: (0, 0, 255),
}

_DOT_POINTS = (
    *ids.LEFT_EYE_EAR,
    *ids.RIGHT_EYE_EAR,
    ids.MOUTH_TOP,
    ids.MOUTH_BOTTOM,
    ids.MOUTH_LEFT,
    ids.MOUTH_RIGHT,
)


def render(
    obs: Observation,
    assessment: Assessment,
    draw_landmarks: bool = True,
    debug: bool = False,
) -> np.ndarray:
    img = obs.frame.image.copy()
    if draw_landmarks and obs.landmarks.detected:
        _draw_points(img, obs.landmarks)
    _draw_panel(img, assessment)
    if assessment.alert_level() >= Level.MEDIUM:
        _draw_alert(img)
    if debug:
        _draw_features(img, obs)
    return img


def draw_calibration(image: np.ndarray, progress: float) -> np.ndarray:
    img = image.copy()
    _text(img, "CALIBRATING", (20, 40), 0.9, (255, 255, 255))
    _text(img, "look at the camera with eyes open", (20, 70), 0.6, (255, 255, 255))
    cv2.rectangle(img, (20, 84), (20 + int(300 * progress), 104), (0, 200, 0), -1)
    cv2.rectangle(img, (20, 84), (320, 104), (220, 220, 220), 1)
    return img


def _text(img: np.ndarray, s: str, org: tuple[int, int], scale: float, color: tuple) -> None:
    # 黒縁を先に描いてから本体を重ねる。明るい背景でも読めるようにするため。
    cv2.putText(img, s, org, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, _FONT, scale, color, 1, cv2.LINE_AA)


def _draw_points(img: np.ndarray, lm: FaceLandmarks) -> None:
    for i in _DOT_POINTS:
        x, y = lm.pixel(i)
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
    if lm.points.shape[0] > ids.RIGHT_IRIS:
        for i in (ids.LEFT_IRIS, ids.RIGHT_IRIS):
            x, y = lm.pixel(i)
            cv2.circle(img, (int(x), int(y)), 2, (255, 0, 0), -1)


def _draw_panel(img: np.ndarray, assessment: Assessment) -> None:
    x, y, step = 16, 40, 66
    for dim in assessment.dimensions.values():
        color = _COLORS[dim.level]
        _text(img, f"{dim.name}: {dim.level.name} {dim.score:.2f}", (x, y), 0.7, color)
        bar_y = y + 12
        cv2.rectangle(img, (x, bar_y), (x + 220, bar_y + 14), (60, 60, 60), -1)
        cv2.rectangle(img, (x, bar_y), (x + int(220 * dim.score), bar_y + 14), color, -1)
        if dim.contributing:
            _text(img, ",".join(dim.contributing), (x, y + 44), 0.5, color)
        y += step


def draw_guided(img: np.ndarray, title: str, instruction: str, phase: str,
                remaining: float, progress: float) -> None:
    # ガイドの指示は上部中央に置く（左上の検知パネルと被らないように）。
    h, w = img.shape[:2]
    from . import jptext

    countdown = f"{remaining:.0f}s"
    (cw, _), _ = cv2.getTextSize(countdown, _FONT, 0.7, 2)
    _text(img, countdown, ((w - cw) // 2, 28), 0.7, (0, 255, 255))
    jptext.put_japanese(img, title, (0, 38), size=30, color=(0, 255, 255), center=True)
    jptext.put_japanese(img, instruction, (0, 84), size=24, color=(255, 255, 255), center=True)

    cv2.rectangle(img, (16, h - 24), (16 + int((w - 32) * progress), h - 12), (0, 200, 0), -1)
    cv2.rectangle(img, (16, h - 24), (w - 16, h - 12), (200, 200, 200), 1)


def draw_record_label(img: np.ndarray, label: str) -> None:
    # 録画中の現在ラベルを右上に大きく出す。ポーズ中のチラ見でも分かるように。
    if not label:
        return
    text = f"REC: {label}"
    (text_w, _), _ = cv2.getTextSize(text, _FONT, 0.8, 2)
    _text(img, text, (img.shape[1] - text_w - 16, 32), 0.8, (0, 0, 255))


def _draw_alert(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 6), (0, 0, 255), -1)
    _text(img, "ALERT", (w - 130, 36), 1.0, (0, 0, 255))


def _draw_features(img: np.ndarray, obs: Observation) -> None:
    # いま使っている特徴量を全部、数値つきで右下に右寄せで出す。
    f = obs.features
    lines = ["-- features --"]
    for key in sorted(f.values):
        lines.append(f"{key}: {f.values[key]:.3f}")
    lines.append(f"ear_base: {obs.profile.ear_open_baseline:.3f}")
    lines.append(f"face: {'yes' if f.face_present else 'no'}")

    h, w = img.shape[:2]
    y0 = h - 16 * len(lines) - 8
    for i, line in enumerate(lines):
        (text_w, _), _ = cv2.getTextSize(line, _FONT, 0.45, 1)
        _text(img, line, (w - text_w - 12, y0 + i * 16), 0.45, (255, 255, 255))
