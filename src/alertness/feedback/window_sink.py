"""OpenCV ウィンドウへの表示。判定結果を映像に重ねて出す出力先。

キー入力やウィンドウを閉じる操作はアプリ側のループで拾う。ここは描画と、
評価軸ごとの警告音だけを担当する（MEDIUM 以上の軸ごとに対応音を鳴らす）。
"""

from __future__ import annotations

from collections.abc import Mapping

import cv2

from ..contracts import Assessment, Observation
from ..labeling import LabelState
from . import display, overlay
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
        alert_min_interval_seconds: float = 1.5,
        alert_escalate_factor: float = 0.7,
        driver_view: bool = False,
        driver_window_width: int = 0,
    ) -> None:
        self._draw_landmarks = draw_landmarks
        self._draw_mesh = draw_mesh
        self._stress_meter = stress_meter
        self._debug = debug
        self._labels = labels  # 録画ラベル表示用（録画中のみ渡される）
        self._alert = AudioAlert(
            alert_cooldown_seconds,
            audio,
            sounds,
            alert_min_interval_seconds,
            alert_escalate_factor,
        )
        self._timeline = self._make_timeline(timeline, timeline_seconds)
        self._window_width = window_width  # 表示の幅。撮影解像度とは切り離してある
        # 運転者向けの別窓。実車では運転者はモニタを見ないので、警告だけの画面を分ける。
        self._driver_view = driver_view
        self._driver_width = driver_window_width or window_width

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
            # シナリオ再生のときだけ levels が入る（通常の収録では空）。
            overlay.draw_expected(image, self._labels.levels)
        display.show(image, self._window_width)
        if self._driver_view:
            from . import driver_view

            size = (image.shape[1], image.shape[0])
            panel = driver_view.render(assessment, size)
            display.show(panel, self._driver_width, driver_view.WINDOW_NAME)
        for dim in assessment.dimensions.values():
            # 収まったことも伝える。伝えないと、次に立ったときに前の警告の続きとして
            # 詰めた間隔から鳴り始める。
            self._alert.trigger(dim.name, dim.level)

    def close(self) -> None:
        cv2.destroyAllWindows()
        display.reset()
