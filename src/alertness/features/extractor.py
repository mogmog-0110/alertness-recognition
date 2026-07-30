"""ランドマークから1フレーム分の生特徴量を計算する FeatureExtractor。"""

from __future__ import annotations

from ..contracts import FaceLandmarks, Features
from ..geometry import euclidean
from . import landmark_ids as ids
from .ear import eye_aspect_ratio
from .gaze import horizontal_gaze_ratio, vertical_gaze_ratio
from .head_pose import estimate_pose
from .mouth import mouth_aspect_ratio

# MediaPipe が出す blendshape の全名称（出力順）。検出器が 52 個すべてを計算しているので、
# 取り出すのはただの転記で追加コストは無い。どれを判定や学習に使うかは後段で選ぶので、
# ここでは選別せず全部通す（映像を捨てたあとで「あの値も欲しかった」を起こさないため）。
# 一部は FACS の AU に対応する: browDown=AU4, eyeSquint=AU7, mouthPress=AU24,
# mouthFrown=AU15, cheekSquint=AU6, browInnerUp=AU1。
# eyeLookDown/Up/In/Out は視線の向きで、幾何の gaze_x / gaze_y と別経路の手がかりになる。
BLENDSHAPE_COLUMNS = (
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
)


class FaceFeatureExtractor:
    """幾何ベースの特徴量を計算する。しきい値判定はしない。"""

    def extract(self, landmarks: FaceLandmarks, timestamp: float) -> Features:
        if not landmarks.detected:
            return Features(values={}, timestamp=timestamp, face_present=False)

        left_ear = eye_aspect_ratio([landmarks.pixel(i) for i in ids.LEFT_EYE_EAR])
        right_ear = eye_aspect_ratio([landmarks.pixel(i) for i in ids.RIGHT_EYE_EAR])
        ear = (left_ear + right_ear) / 2.0

        mar = mouth_aspect_ratio(
            landmarks.pixel(ids.MOUTH_TOP),
            landmarks.pixel(ids.MOUTH_BOTTOM),
            landmarks.pixel(ids.MOUTH_LEFT),
            landmarks.pixel(ids.MOUTH_RIGHT),
        )
        pose = estimate_pose(landmarks)
        face_scale = euclidean(
            landmarks.pixel(ids.LEFT_EYE_OUTER), landmarks.pixel(ids.RIGHT_EYE_OUTER)
        )
        # 画面内での頭の位置。正規化座標(0..1)なので撮影解像度に依らない。face_scale が
        # 奥行き方向を表すのに対し、こちらは上下左右。座席で沈み込む・寄りかかる動きが出る。
        head_x, head_y = (float(v) for v in landmarks.points[ids.NOSE_TIP, :2])

        values: dict[str, float] = {
            "ear": ear,
            "ear_left": left_ear,
            "ear_right": right_ear,
            "mar": mar,
            "pitch": pose.pitch,
            "yaw": pose.yaw,
            "roll": pose.roll,
            "gaze_x": horizontal_gaze_ratio(landmarks),
            "gaze_y": vertical_gaze_ratio(landmarks),
            "head_x": head_x,
            "head_y": head_y,
            "face_scale": face_scale,
        }
        for key in BLENDSHAPE_COLUMNS:
            if key in landmarks.blendshapes:
                values[key] = float(landmarks.blendshapes[key])
        return Features(values=values, timestamp=timestamp, face_present=True)
