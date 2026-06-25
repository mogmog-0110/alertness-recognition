"""層と層をつなぐ口（ポート）の定義。

ここでは振る舞いだけを Protocol で決め、具体的な実装（OpenCV や MediaPipe、
ルールや機械学習）には依存しない。パイプラインはこのポートにのみ依存するので、
実装を差し替えても本体は無修正で済む。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from .contracts import (
    Assessment,
    CalibrationProfile,
    CueResult,
    FaceLandmarks,
    Features,
    Frame,
    Observation,
)


@runtime_checkable
class FrameSource(Protocol):
    """映像の入力源。PCカメラ・動画ファイル・将来のモバイル映像など。"""

    def frames(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...


@runtime_checkable
class LandmarkDetector(Protocol):
    """画像から顔ランドマークを取り出す。"""

    def detect(self, frame: Frame) -> FaceLandmarks: ...

    def close(self) -> None: ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """ランドマークから生の特徴量を計算する。しきい値判定はしない。"""

    def extract(self, landmarks: FaceLandmarks, timestamp: float) -> Features: ...


@runtime_checkable
class Cue(Protocol):
    """特徴ごとの判定器。1つの評価軸に対する弱い手がかりを出す。"""

    name: str
    dimension: str

    def evaluate(self, obs: Observation) -> CueResult: ...


@runtime_checkable
class DecisionPolicy(Protocol):
    """cue の結果を統合して最終判定を作る方針。ルール／機械学習など。"""

    def decide(self, obs: Observation, cues: Sequence[CueResult]) -> Assessment: ...


@runtime_checkable
class Classifier(Protocol):
    """判定層の安定した口。ルールベースも ML/DL も、すべてこれを満たす。"""

    def assess(self, obs: Observation) -> Assessment: ...


@runtime_checkable
class Calibrator(Protocol):
    """起動時などに基準値（プロファイル）を採取する。"""

    def collect(self, obs: Observation) -> None: ...

    def finalize(self) -> CalibrationProfile: ...

    @property
    def progress(self) -> float:  # 0..1
        ...


@runtime_checkable
class FeedbackSink(Protocol):
    """判定結果の出力先。画面表示・録画・通知など。複数を束ねてもよい。"""

    def emit(self, obs: Observation, assessment: Assessment) -> None: ...

    def close(self) -> None: ...
