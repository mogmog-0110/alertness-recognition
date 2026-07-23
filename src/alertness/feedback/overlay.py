"""映像へのオーバーレイ描画。

画面上の文字は OpenCV の制約で日本語を描けないため英字表記にしている
（コメントやログは日本語）。どんな背景でも読めるよう黒縁付きで描く。
debug=True のときは生の特徴量も出し、しきい値調整の手がかりにする。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..contracts import Assessment, FaceLandmarks, Level, Observation
from ..features import landmark_ids as ids
from ..features.rppg import forehead_roi_box

WINDOW_NAME = "Alertness"
_PANEL_WIDTH = 190  # debug 表示の板の幅（右下）
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

# 顔全体のメッシュ（テッセレーション）の辺。mediapipe から一度だけ取り込む。
_TESSELLATION: tuple[tuple[int, int], ...] | None = None
_TESS_LOADED = False


def _tessellation() -> tuple[tuple[int, int], ...] | None:
    global _TESSELLATION, _TESS_LOADED
    if not _TESS_LOADED:
        _TESS_LOADED = True
        try:
            from mediapipe.solutions.face_mesh import (  # type: ignore[import]
                FACEMESH_TESSELLATION,
            )

            _TESSELLATION = tuple(FACEMESH_TESSELLATION)
        except Exception:
            _TESSELLATION = None  # mediapipe 無し等。全点ドットで代替する。
    return _TESSELLATION


def render(
    obs: Observation,
    assessment: Assessment,
    draw_landmarks: bool = True,
    debug: bool = False,
    draw_mesh: bool = False,
    stress_meter: bool = False,
) -> np.ndarray:
    img = obs.frame.image.copy()
    if obs.landmarks.detected:
        if draw_mesh:
            _draw_mesh(img, obs.landmarks)
        elif draw_landmarks:
            _draw_points(img, obs.landmarks)
    _draw_panel(img, assessment)
    if stress_meter:
        if obs.landmarks.detected:
            _draw_rppg_roi(img, obs.landmarks, assessment)  # ストレス＝額のどこを測っているか
        _draw_stress_meter(img, obs, assessment)
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


def _draw_mesh(img: np.ndarray, lm: FaceLandmarks) -> None:
    # 顔全体のメッシュ。辺が取れれば網目を、無ければ全点をドットで描く。
    n = lm.points.shape[0]
    conns = _tessellation()
    if conns:
        for a, b in conns:
            if a < n and b < n:
                xa, ya = lm.pixel(a)
                xb, yb = lm.pixel(b)
                cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (0, 160, 0), 1)
    else:
        for i in range(n):
            x, y = lm.pixel(i)
            cv2.circle(img, (int(x), int(y)), 1, (0, 160, 0), -1)


def _draw_panel(img: np.ndarray, assessment: Assessment) -> None:
    # バーの長さと段階は「警告の強さ」。集中のように高いほど良い軸は反転して表示される
    # （inattentive が伸びる＝集中していない）ので、元のスコアも併記する。
    x, y, step = 16, 40, 66
    for dim in assessment.dimensions.values():
        color = _COLORS[dim.level]
        _text(img, f"{dim.display_name}: {dim.level.name} {dim.alarm:.2f}", (x, y), 0.7, color)
        bar_y = y + 12
        cv2.rectangle(img, (x, bar_y), (x + 220, bar_y + 14), (60, 60, 60), -1)
        cv2.rectangle(img, (x, bar_y), (x + int(220 * dim.alarm), bar_y + 14), color, -1)
        notes = list(dim.contributing)
        if dim.alert_score is not None:
            notes.insert(0, f"{dim.name} {dim.score:.2f}")
        if notes:
            _text(img, ",".join(notes), (x, y + 44), 0.5, color)
        y += step


def _draw_rppg_roi(img: np.ndarray, lm: FaceLandmarks, assessment: Assessment) -> None:
    # 額のどこで心拍を測っているかを、いまのストレス段階の色で塗る。
    # 面の中に色の分布は作らない。rPPG は額全体の平均しか見ておらず、場所ごとの差は
    # 持っていないので、濃淡を付けると無い解像度があるように見えてしまう。
    h, w = img.shape[:2]
    box = forehead_roi_box(lm, w, h)
    if box is None:
        return
    x0, y0, x1, y1 = box
    dim = assessment.dimensions.get("stress")
    cue = next((c for c in assessment.cues if c.dimension == "stress"), None)
    measuring = cue is None or cue.valid
    color = _COLORS[dim.level] if (dim is not None and measuring) else (150, 150, 150)

    roi = img[y0:y1, x0:x1]
    if roi.size:
        tint = np.full(roi.shape, color, dtype=np.uint8)
        img[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.65, tint, 0.35, 0)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
    label = "stress" if measuring else "stress (measuring)"
    _text(img, label, (x0, max(12, y0 - 4)), 0.4, color)


def _draw_stress_meter(img: np.ndarray, obs: Observation, assessment: Assessment) -> None:
    # 右上に、ストレスの安静基準キャリブの進行度を出す（起動時キャリブと同じ見た目のバー）。
    cue = next((c for c in assessment.cues if c.name == "hr_elevation"), None)
    if cue is None or cue.progress is None:
        return  # ストレス cue が無い/較正不要（固定基準）のときは出さない。

    progress = max(0.0, min(1.0, cue.progress))
    done = progress >= 1.0
    color = (0, 200, 0) if done else (0, 180, 220)

    w = img.shape[1]
    x, y, bw = w - 236, 60, 220
    _text(img, "STRESS CALIB", (x, y), 0.55, (255, 255, 255))
    bar_y = y + 12
    cv2.rectangle(img, (x, bar_y), (x + int(bw * progress), bar_y + 14), color, -1)
    cv2.rectangle(img, (x, bar_y), (x + bw, bar_y + 14), (220, 220, 220), 1)
    label = "calibrated" if done else f"{progress * 100:.0f}%"
    _text(img, label, (x, bar_y + 34), 0.5, color)

    # 0% から進まないときの理由。HRが出ていなければ rPPG 無効/未取得、出ているのに
    # 進まなければ信号品質が閾値未満（照明・動きを見直す合図）。
    if not done and progress <= 0.0:
        hr = obs.features.get("hr_bpm", float("nan"))
        hint = "low signal" if not math.isnan(hr) else "no rPPG signal"
        _text(img, hint, (x, bar_y + 52), 0.42, (0, 170, 255))


def draw_guided(
    img: np.ndarray, title: str, instruction: str, phase: str, remaining: float, progress: float
) -> None:
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
    # いま使っている特徴量を全部、数値つきで右下に出す。
    # 文字は縁取りせず、後ろに半透明の板を敷いて読ませる。特徴量が増えて行数が
    # 3倍近くになり、1行2回の描画（縁取り＋本体）だけで 18ms/フレーム 掛かっていたため。
    f = obs.features
    lines = ["-- features --"]
    lines += [f"{key}: {f.values[key]:.3f}" for key in sorted(f.values)]
    lines.append(f"ear_base: {obs.profile.ear_open_baseline:.3f}")
    lines.append(f"face: {'yes' if f.face_present else 'no'}")
    measured = getattr(obs.history, "measured_fps", None)
    if measured:
        lines.append(f"fps: {measured:.1f}")  # 要求値ではなく実際に流れている値

    h, w = img.shape[:2]
    step = 16
    y0 = h - step * len(lines) - 8
    x0 = w - _PANEL_WIDTH - 12
    panel = img[max(0, y0 - 14) : h, max(0, x0 - 8) : w]
    if panel.size:
        panel[:] = cv2.addWeighted(panel, 0.35, np.zeros_like(panel), 0.65, 0)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x0, y0 + i * step), _FONT, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
