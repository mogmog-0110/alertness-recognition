"""MediaPipe FaceLandmarker（Tasks API）による顔ランドマーク検出。

478点のランドマークと、あくび・瞬きの判定に使う blendshape を返す。
動画モードなので、フレームのタイムスタンプは単調増加で渡す必要がある。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from ..contracts import FaceLandmarks, Frame


class MediaPipeDetector:
    def __init__(
        self, model_path: str, max_faces: int = 1, output_blendshapes: bool = True
    ) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"モデルが見つかりません: {model_path}\n"
                "scripts\\setup.bat を実行してダウンロードしてください。"
            )
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            output_face_blendshapes=output_blendshapes,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame: Frame) -> FaceLandmarks:
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, int(frame.timestamp * 1000))

        w, h = frame.size
        if not result.face_landmarks:
            return FaceLandmarks(points=np.zeros((0, 3)), image_size=(w, h), detected=False)

        points = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=float)
        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            for cat in result.face_blendshapes[0]:
                blendshapes[cat.category_name] = float(cat.score)
        return FaceLandmarks(
            points=points, image_size=(w, h), detected=True, blendshapes=blendshapes
        )

    def close(self) -> None:
        self._landmarker.close()
