"""統計的キャリブレーション。数秒ぶんの特徴量を集めて基準値を作る。

起動直後の「楽な姿勢で正面・開眼」を基準にする。次の2点で頑健にしている:
- カメラ起動直後のウォームアップ frame は捨てる（露出調整中で値が暴れるため）。
- 平均ではなく中央値を使う（瞬きなどの外れ値に引っ張られないため）。

幾何（EAR・姿勢・視線）の基準は数秒で取れるが、rPPG の心拍は窓（既定20秒）が満ちるまで
出ない。安静中心化(ML経路)は心拍もその人の安静からの差にしたいので、心拍を必要とする
ときは、心拍が現れるまでキャリブを延ばせるようにする（キャリブ中はずっと安静なので、
そこで取れた心拍がそのまま安静基準になる）。心拍が出ないまま上限に達したら諦めて確定する。
"""

from __future__ import annotations

import math

import numpy as np

from ..contracts import CalibrationProfile, Features, Observation, Pose


class StatisticalCalibrator:
    def __init__(
        self,
        duration_seconds: float = 3.0,
        fps: float = 30.0,
        warmup_seconds: float = 0.7,
        require_keys: tuple[str, ...] = (),
        max_seconds: float = 0.0,
    ) -> None:
        self._needed = max(5, int(duration_seconds * fps))
        self._warmup = int(warmup_seconds * fps)
        # 心拍など、遅れて出る特徴を安静基準に含めたいときに指定する。出そろうまで確定を待つ。
        self._require = tuple(require_keys)
        self._min_key = max(3, int(0.5 * fps))  # 必要な特徴を安定と見なす最小サンプル数
        # 待つ上限。心拍が出ないカメラ/環境で永遠に待たないための打ち切り。0 なら待たない。
        self._max = int(max_seconds * fps) if max_seconds > 0 else 0
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
        if self._max and len(self._samples) >= self._max:
            return 1.0  # 上限に達した。必要な特徴が出そろわなくても確定する。
        frame_ratio = len(self._samples) / self._needed
        if not self._require:
            return min(1.0, frame_ratio)
        # 幾何が揃い、かつ要求した特徴（心拍など）も最小数そろって初めて完了。
        key_ratio = min(self._key_count(key) / self._min_key for key in self._require)
        return min(1.0, frame_ratio, key_ratio)

    def _key_count(self, key: str) -> int:
        return sum(1 for f in self._samples if not math.isnan(f.get(key)))

    def finalize(self) -> CalibrationProfile:
        if not self._samples:
            # 1サンプルも取れなかったときは中立プロファイルで素通しにする。
            return CalibrationProfile.identity()
        return CalibrationProfile(
            ear_open_baseline=self._median("ear", 0.3),
            mar_neutral=self._median("mar", 0.0),
            head_pose_neutral=Pose(
                self._angle_median("pitch", 0.0),
                self._angle_median("yaw", 0.0),
                self._angle_median("roll", 0.0),
            ),
            gaze_center=(self._median("gaze_x", 0.5), self._median("gaze_y", 0.5)),
            face_scale=self._median("face_scale", 1.0),
            baselines=self._baselines(),
        )

    def _angle_median(self, key: str, default: float) -> float:
        """角度の中央値。±180 を跨いでも壊れないように円周上で取る。

        角度は循環量なので、そのまま median を取ってはいけない。solvePnP が返す
        pitch は正面を向いていても ±180 付近に居座ることがあり（3D モデルの Y 軸が
        上向きで画像の Y 軸が下向きなので 180 度のオフセットが乗る）、標本が
        +179 と -179 に割れる。普通の中央値だと基準が 0 付近になり、以後ずっと
        「下を向いている」と判定され続ける（実測: 正面を向いたまま pitch_rel が
        22 度ずれ、head_down が立ちっぱなしになった）。

        単位ベクトルの平均で中心を出し、その周りへ畳んでから中央値を取る。
        平均ではなく中央値なのは、キャリブ中の一瞬の検出ミスに引きずられない
        ようにするため（元の実装の意図をそのまま保つ）。
        """
        values = [f.get(key) for f in self._samples]
        values = [v for v in values if not math.isnan(v)]
        if not values:
            return default
        radians = np.radians(values)
        center = math.atan2(float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians))))
        offsets = np.angle(np.exp(1j * (radians - center)))
        median = center + float(np.median(offsets))
        return float((math.degrees(median) + 180.0) % 360.0 - 180.0)

    def _median(self, key: str, default: float) -> float:
        values = [f.get(key) for f in self._samples]
        values = [v for v in values if not math.isnan(v)]
        return float(np.median(values)) if values else default

    def _baselines(self) -> dict[str, float]:
        # 安静時に観測できた特徴量ごとの中央値。ML経路が「その人の平常からの差」を取るのに
        # 使う。キャリブ中に一度でも値が出た特徴だけを持つ（hr は起動直後は出ないことが多い）。
        keys = {k for f in self._samples for k in f.values}
        baselines = {}
        for key in keys:
            values = [f.get(key) for f in self._samples if not math.isnan(f.get(key))]
            if values:
                baselines[key] = float(np.median(values))
        return baselines
