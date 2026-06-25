"""統計的キャリブレーション。数秒ぶんの特徴量を集めて基準値を作る。

起動直後の「楽な姿勢で正面・開眼」を基準にする。次の2点で頑健にしている:
- カメラ起動直後のウォームアップ frame は捨てる（露出調整中で値が暴れるため）。
- 平均ではなく中央値を使う（瞬きなどの外れ値に引っ張られないため）。
"""

from __future__ import annotations

import math

import numpy as np

from ..contracts import CalibrationProfile, Features, Observation, Pose


class StatisticalCalibrator:
    def __init__(
        self, duration_seconds: float = 3.0, fps: float = 30.0, warmup_seconds: float = 0.7
    ) -> None:
        self._needed = max(5, int(duration_seconds * fps))
        self._warmup = int(warmup_seconds * fps)
        self._seen = 0
        self._samples: list[Features] = []

    def collect(self, obs: Observation) -> None:
        if not obs.features.face_present:
            return
        self._seen += 1
        if self._seen <= self._warmup:  # ウォームアップ分は基準に使わない
            return
        self._samples.append(obs.features)

    @property
    def progress(self) -> float:
        return min(1.0, len(self._samples) / self._needed)

    def finalize(self) -> CalibrationProfile:
        if not self._samples:
            # 1サンプルも取れなかったときは中立プロファイルで素通しにする。
            return CalibrationProfile.identity()
        return CalibrationProfile(
            ear_open_baseline=self._median("ear", 0.3),
            mar_neutral=self._median("mar", 0.0),
            head_pose_neutral=Pose(
                self._median("pitch", 0.0),
                self._median("yaw", 0.0),
                self._median("roll", 0.0),
            ),
            gaze_center=(self._median("gaze_x", 0.5), 0.5),
            face_scale=self._median("face_scale", 1.0),
        )

    def _median(self, key: str, default: float) -> float:
        values = [f.get(key) for f in self._samples]
        values = [v for v in values if not math.isnan(v)]
        return float(np.median(values)) if values else default
