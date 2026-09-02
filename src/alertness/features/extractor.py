"""ランドマークから1フレーム分の生特徴量を計算する FeatureExtractor。"""

from __future__ import annotations

from ..contracts import FaceLandmarks, Features, Pose
from ..geometry import euclidean
from . import landmark_ids as ids
from .ear import eye_aspect_ratio
from .gaze import horizontal_gaze_ratio, vertical_gaze_ratio
from .head_pose import _wrap as _wrap_deg
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

    def __init__(self) -> None:
        # 直前に採用した姿勢とその時刻。ありえない飛びを捨てるのに使う。
        self._last_pose: tuple[Pose | None, float] = (None, 0.0)

    # 頭が実際に動ける速さの上限 (度/秒)。素早く振り向く動作でも 300 度/秒には
    # 届かない。これを超える変化は solvePnP の反転解や検出のちらつきであって、
    # 実際の動きではない。
    MAX_POSE_RATE_DEG_PER_SEC = 300.0

    def _stable_pose(self, landmarks: FaceLandmarks, timestamp: float) -> Pose:
        """物理的にありえない姿勢の飛びを捨てる。

        solvePnP は時折もう一方の解へ飛び、pitch が 1 フレームで 100 度以上
        変わることがある (実測: 30fps で 40 度以上の飛びが 62 回)。1 フレームの
        飛びでも、うなずき判定は「8 度以上の上下動」を数えるので偽のうなずきが
        立ち、60 秒の窓のあいだ眠気が最大に張り付く。

        直前に**採用した**姿勢と、その時刻からの経過で許容量を決める。捨てても
        時刻を進めないので、許容量は時間とともに広がり、必ず復帰する。

        捨てるたびに時刻を進めると許容量が 1 フレーム分のまま増えず、いったん
        実際の姿勢から離れると永久に古い値を返し続ける (実測: yaw が 54.3 度で
        固まり、覚醒フレームの 44% がその値になって head_turn が誤発火した)。
        """
        pose = estimate_pose(landmarks)
        previous, previous_at = self._last_pose
        if previous is None or timestamp <= previous_at:
            self._last_pose = (pose, timestamp)
            return pose
        elapsed = timestamp - previous_at
        limit = self.MAX_POSE_RATE_DEG_PER_SEC * elapsed
        moved = max(
            abs(_wrap_deg(pose.pitch - previous.pitch)),
            abs(_wrap_deg(pose.yaw - previous.yaw)),
            abs(_wrap_deg(pose.roll - previous.roll)),
        )
        if moved > limit:
            # 捨てる。**時刻は進めない**。次のフレームでは許容量が広がるので、
            # 本当にその姿勢なら数フレームで追いつく。300 度/秒なら 0.5 秒後に
            # 150 度まで許すので、どんな姿勢からでも必ず復帰する。
            return previous
        self._last_pose = (pose, timestamp)
        return pose

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
        pose = self._stable_pose(landmarks, timestamp)
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
