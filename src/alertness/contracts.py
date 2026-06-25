"""層をまたいで受け渡す不変データの定義。

各層（入力源・検出器・特徴量・判定・出力）の境界をここに集約する。
すべて frozen（不変）なので、生成後は作り変えず、新しい値を作って渡す。
これらは数値の塊なので、ネットワークやファイルを越えても運べる
（＝モバイル分割や Colab 学習との接続点になる）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Frame:
    """カメラ1枚分の映像。"""

    image: np.ndarray  # OpenCV の BGR 画像
    index: int
    timestamp: float  # 秒。単調増加であればよい
    source_id: str = "camera"

    @property
    def size(self) -> tuple[int, int]:
        # (幅, 高さ)
        h, w = self.image.shape[:2]
        return (int(w), int(h))


@dataclass(frozen=True)
class FaceLandmarks:
    """検出した顔の点群。検出失敗時は detected=False で空の点群を持つ。"""

    points: np.ndarray  # (N, 3) 正規化座標。x,y は 0..1、z は相対深度
    image_size: tuple[int, int]  # (幅, 高さ)
    detected: bool
    blendshapes: Mapping[str, float] = field(default_factory=dict)

    def pixel(self, i: int) -> tuple[float, float]:
        # 正規化座標を画素座標に変換する
        w, h = self.image_size
        return (float(self.points[i, 0]) * w, float(self.points[i, 1]) * h)


@dataclass(frozen=True)
class Features:
    """1フレームから計算した特徴量。生の数値だけを持ち、判定はしない。

    値は名前付きの数値（ear, mar, pitch ...）。正規化済みの値も同じ袋に
    入れる（例: ear_norm）。学習時はこの袋がそのまま入力ベクトルになる。
    """

    values: Mapping[str, float]
    timestamp: float
    face_present: bool = True

    def get(self, name: str, default: float = float("nan")) -> float:
        return float(self.values.get(name, default))


class History(Protocol):
    """直近フレームの特徴量履歴を読むためのビュー。

    PERCLOS や瞬きなど時系列を要する判定は、ここから過去フレームを取り出す。
    将来 LSTM などが時系列窓を読む口にもなる。
    """

    @property
    def fps(self) -> float: ...

    def latest(self) -> Features | None: ...

    def recent(self, seconds: float) -> Sequence[Features]: ...


@dataclass(frozen=True)
class Pose:
    """頭部の向き（度）。"""

    pitch: float  # 下向きが正、上向きが負
    yaw: float  # 右向き/左向き
    roll: float  # 首のかしげ


@dataclass(frozen=True)
class CalibrationProfile:
    """その人・その設置環境での基準値。判定を相対化するために使う。

    学習と推論で同じ正規化を使うため schema_version を持たせ、取り違えを防ぐ。
    """

    ear_open_baseline: float  # 楽に開けたときの EAR
    mar_neutral: float  # 口を閉じたときの MAR
    head_pose_neutral: Pose  # 正面・中立時の頭部姿勢
    gaze_center: tuple[float, float]  # 画面中心を見たときの虹彩位置（正規化）
    face_scale: float  # 目間距離。距離正規化の基準
    user_id: str = "default"
    schema_version: int = 1

    @staticmethod
    def identity() -> CalibrationProfile:
        # キャリブ前に使う中立プロファイル。正規化をほぼ素通しにする。
        return CalibrationProfile(
            ear_open_baseline=0.3,
            mar_neutral=0.0,
            head_pose_neutral=Pose(0.0, 0.0, 0.0),
            gaze_center=(0.5, 0.5),
            face_scale=1.0,
        )


@dataclass(frozen=True)
class Observation:
    """ある瞬間に分かっていることを1つに束ねた、判定層への入力。

    判定の実装は、必要なものだけここから読む
    （ルール=cue結果, SVM=features, LSTM=history, CNN=frame）。
    """

    frame: Frame
    landmarks: FaceLandmarks
    features: Features
    history: History
    profile: CalibrationProfile


class Level(IntEnum):
    """評価軸の段階。二値判定なら NONE/HIGH だけ使う、という退化もできる。"""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass(frozen=True)
class CueResult:
    """特徴ごとの判定結果。どの評価軸に、どれだけ効くかを表す。"""

    name: str  # cue 名（"eye_closure" など）
    dimension: str  # 効く評価軸（"drowsiness" など）
    score: float  # 0..1 の重症度
    active: bool  # 自分の閾値を超えたか
    detail: str = ""  # 画面表示・根拠用の短い説明


@dataclass(frozen=True)
class Dimension:
    """1本の評価軸の結果。"""

    name: str
    score: float  # 0..1
    level: Level
    contributing: tuple[str, ...] = ()  # 根拠になった cue 名


@dataclass(frozen=True)
class Assessment:
    """最終判定。何を判定するかを「評価軸の集合」として表す。

    眠気と注意逸脱は別の軸なので、同時に立つこともある。
    cues には各 cue の結果も持たせ、どの手がかりが効いたかを後から調べられる。
    """

    dimensions: Mapping[str, Dimension]
    timestamp: float
    cues: tuple[CueResult, ...] = ()

    def alert_level(self) -> Level:
        # 全軸のうち最も高いレベル。フィードバックの強さ決めに使う。
        if not self.dimensions:
            return Level.NONE
        return max(d.level for d in self.dimensions.values())

    def headline(self) -> Dimension | None:
        # 最も重症の軸。簡易表示用。
        if not self.dimensions:
            return None
        return max(self.dimensions.values(), key=lambda d: d.score)
