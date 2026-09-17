"""MediaPipe FaceLandmarker（Tasks API）による顔ランドマーク検出。

478点のランドマークと、あくび・瞬きの判定に使う blendshape を返す。
動画モードなので、フレームのタイムスタンプは単調増加で渡す必要がある。
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2
import mediapipe as mp  # type: ignore[import]
import numpy as np
from mediapipe.tasks import python as mp_python  # type: ignore[import]
from mediapipe.tasks.python import vision  # type: ignore[import]

from .. import log
from ..contracts import FaceLandmarks, Frame


@contextmanager
def _native_stderr_silenced() -> Iterator[None]:
    """C++ 側が標準エラーへ直接書くログを、この間だけ捨てる。

    モデルの読み込みで XNNPACK や feedback manager の W/INFO 行が 4 行出るが、
    利用者が対応できることは何もない。Python の sys.stderr を差し替えても
    ネイティブの書き込みは止まらないので、fd 2 そのものを付け替える。
    """
    if log.is_verbose():
        yield
        return
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


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
        with _native_stderr_silenced():
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
