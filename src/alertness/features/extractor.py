"""ランドマークから1フレーム分の生特徴量を計算する FeatureExtractor。"""

from __future__ import annotations

from ..contracts import FaceLandmarks, Features
from ..geometry import euclidean
from . import landmark_ids as ids
from .ear import eye_aspect_ratio
from .gaze import horizontal_gaze_ratio
from .head_pose import estimate_pose
from .mouth import mouth_aspect_ratio

# blendshape から取り込む値（あくび・瞬きの補助）
_BLENDSHAPES = ("jawOpen", "eyeBlinkLeft", "eyeBlinkRight")


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
        gaze_x = horizontal_gaze_ratio(landmarks)
        face_scale = euclidean(
            landmarks.pixel(ids.LEFT_EYE_OUTER), landmarks.pixel(ids.RIGHT_EYE_OUTER)
        )

        values: dict[str, float] = {
            "ear": ear,
            "ear_left": left_ear,
            "ear_right": right_ear,
            "mar": mar,
            "pitch": pose.pitch,
            "yaw": pose.yaw,
            "roll": pose.roll,
            "gaze_x": gaze_x,
            "face_scale": face_scale,
        }
        for key in _BLENDSHAPES:
            if key in landmarks.blendshapes:
                values[key] = float(landmarks.blendshapes[key])
        return Features(values=values, timestamp=timestamp, face_present=True)
