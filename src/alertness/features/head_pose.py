"""頭部姿勢（pitch/yaw/roll）の推定。

汎用の3D顔モデルとランドマークの対応から solvePnP で姿勢を求める。
カメラ内部パラメータは未校正なので焦点距離を画像幅で近似する。絶対角度は
粗いが、キャリブで中立姿勢を基準に相対化するため判定には十分。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..contracts import FaceLandmarks, Pose
from . import landmark_ids as ids

# 代表点の汎用3Dモデル座標（おおよその mm）。下の image_points と順序を合わせる。
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # 鼻先
        (0.0, -63.6, -12.5),  # あご
        (-43.3, 32.7, -26.0),  # 左目尻
        (43.3, 32.7, -26.0),  # 右目尻
        (-28.9, -28.9, -24.1),  # 左口角
        (28.9, -28.9, -24.1),  # 右口角
    ],
    dtype=float,
)


def estimate_pose(landmarks: FaceLandmarks) -> Pose:
    w, h = landmarks.image_size
    image_points = np.array(
        [
            landmarks.pixel(ids.NOSE_TIP),
            landmarks.pixel(ids.CHIN),
            landmarks.pixel(ids.LEFT_EYE_CORNER),
            landmarks.pixel(ids.RIGHT_EYE_CORNER),
            landmarks.pixel(ids.LEFT_MOUTH_CORNER),
            landmarks.pixel(ids.RIGHT_MOUTH_CORNER),
        ],
        dtype=float,
    )
    focal = float(w)
    camera_matrix = np.array(
        [[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]],
        dtype=float,
    )
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS, image_points, camera_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return Pose(0.0, 0.0, 0.0)
    rmat, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = _rotation_to_euler(rmat)
    # solvePnP は時々 ±180/360 付近に飛ぶ。[-180,180] に畳んで暴走値をならす
    # （例: 357° → -3°）。これをしないと head_down cue が誤発火する。
    return Pose(pitch=_wrap(pitch), yaw=_wrap(yaw), roll=_wrap(roll))


def _wrap(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def _rotation_to_euler(r: np.ndarray) -> tuple[float, float, float]:
    # 回転行列をオイラー角（度）に分解する。pitch=下向き正の近似。
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0.0
    else:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    return math.degrees(x), math.degrees(y), math.degrees(z)
