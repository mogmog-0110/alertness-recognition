"""OpenCV ウィンドウへの表示。判定結果を映像に重ねて出す出力先。

キー入力やウィンドウを閉じる操作はアプリ側のループで拾う。ここは描画と、
評価軸ごとの警告音だけを担当する（MEDIUM 以上の軸ごとに対応音を鳴らす）。
"""

from __future__ import annotations

from collections.abc import Mapping

import cv2

from ..contracts import Assessment, Level, Observation
from ..labeling import LabelState
from . import overlay
from .alert import AudioAlert


class OpenCvWindowSink:
    def __init__(
        self,
        draw_landmarks: bool = True,
        audio: bool = True,
        alert_cooldown_seconds: float = 5.0,
        debug: bool = False,
        sounds: Mapping[str, str] | None = None,
        labels: LabelState | None = None,
        draw_mesh: bool = False,
        stress_meter: bool = False,
        timeline: str = "",
        timeline_seconds: float = 300.0,
        window_width: int = 0,
    ) -> None:
        self._draw_landmarks = draw_landmarks
        self._draw_mesh = draw_mesh
        self._stress_meter = stress_meter
        self._debug = debug
        self._labels = labels  # 録画ラベル表示用（録画中のみ渡される）
        self._alert = AudioAlert(alert_cooldown_seconds, audio, sounds)
        self._timeline = self._make_timeline(timeline, timeline_seconds)
        # 表示の大きさは撮影解像度と切り離す。rPPG の精度は実効フレームレートで決まるので
        # 撮影は軽いままにし、見づらいときはここで拡大する。
        self._window_width = window_width

    @staticmethod
    def _make_timeline(name: str, seconds: float):
        if not name:
            return None
        from .timeline import DimensionTimeline

        return DimensionTimeline(name, span_seconds=seconds)

    def emit(self, obs: Observation, assessment: Assessment) -> None:
        image = overlay.render(
            obs,
            assessment,
            self._draw_landmarks,
            self._debug,
            self._draw_mesh,
            self._stress_meter,
        )
        if self._timeline is not None:
            self._timeline.render(image, assessment)
        if self._labels is not None:
            overlay.draw_record_label(image, self._labels.value)
        cv2.imshow(overlay.WINDOW_NAME, self._fit(image))
        for dim in assessment.dimensions.values():
            if dim.level >= Level.MEDIUM:
                self._alert.trigger(dim.name)

    def _fit(self, image):
        # 縦横比は保ったまま、指定の幅に合わせる。0 なら撮影したままの大きさで出す。
        width = self._window_width
        if width <= 0 or width == image.shape[1]:
            return image
        height = round(image.shape[0] * width / image.shape[1])
        interpolation = cv2.INTER_LINEAR if width > image.shape[1] else cv2.INTER_AREA
        return cv2.resize(image, (width, height), interpolation=interpolation)

    def close(self) -> None:
        cv2.destroyAllWindows()
