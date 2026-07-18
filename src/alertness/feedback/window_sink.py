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
        rppg_view: bool = False,
        fps: float = 30.0,
    ) -> None:
        self._draw_landmarks = draw_landmarks
        self._draw_mesh = draw_mesh
        self._stress_meter = stress_meter
        self._debug = debug
        self._labels = labels  # 録画ラベル表示用（録画中のみ渡される）
        self._alert = AudioAlert(alert_cooldown_seconds, audio, sounds)
        self._rppg_view = self._make_rppg_view(fps) if rppg_view else None

    @staticmethod
    def _make_rppg_view(fps: float):
        from .rppg_view import RppgView

        return RppgView(fps=fps)

    def emit(self, obs: Observation, assessment: Assessment) -> None:
        image = overlay.render(
            obs,
            assessment,
            self._draw_landmarks,
            self._debug,
            self._draw_mesh,
            self._stress_meter,
        )
        if self._rppg_view is not None:
            self._rppg_view.render(image, obs)
        if self._labels is not None:
            overlay.draw_record_label(image, self._labels.value)
        cv2.imshow(overlay.WINDOW_NAME, image)
        for dim in assessment.dimensions.values():
            if dim.level >= Level.MEDIUM:
                self._alert.trigger(dim.name)

    def close(self) -> None:
        cv2.destroyAllWindows()
