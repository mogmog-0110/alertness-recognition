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
class Resettable(Protocol):
    """溜めた状態を捨てられる部品。cue の安静基準や注意残高など。

    運転者が交代すると、前の人の基準や残高は次の人に対して誤りになる。
    再キャリブレーション時にこの口を通してまとめて捨てる。状態を持たない
    実装は満たさなくてよく、呼び出し側が isinstance で選り分ける。
    """

    def reset(self) -> None: ...


@runtime_checkable
class Interruptible(Protocol):
    """待ちを外から打ち切れる入力源。

    ネットワーク越しの入力は、繋がっていない間 frames() が何も返さずに待ち続ける。
    画面もキーも無い運転では SIGTERM で終わらせるしかないが、待ちに入ったままでは
    終了の合図に気づけない。この口で待ちを解いてからループを畳む。
    """

    def interrupt(self) -> None: ...


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


@runtime_checkable
class RemoteControlled(Protocol):
    """離れた端末から操作を受け取れる入力源。

    端末は運転者の手元にあり、PC の画面もキーボードも触れない。基準を取り直す
    手段が PC 側のキー操作しか無いと、実験のたびにプロセスを落とすことになる。

    実装は任意。この口を持たない source には呼ばれない。
    """

    def take_commands(self) -> list[str]: ...


@runtime_checkable
class CalibrationAware(Protocol):
    """キャリブ中の進み具合も受け取れる出力先。

    emit は判定が出てから呼ばれるので、キャリブ中は何も届かない。手元に画面のある
    PC なら窓へ進捗を描けばよいが、端末が離れた場所にある構成では「いま基準を
    測っている」ことが利用者に伝わらない。伝わらないと正面から目を離してしまい、
    基準そのものが狂う。

    実装は任意。この口を持たない sink には呼ばれない。
    """

    def calibrating(
        self, obs: Observation, progress: float,
        waiting_for: str = "", expected_seconds: float = 0.0,
    ) -> None: ...
