"""1フレームの処理をまとめる。ポートにのみ依存し、具体実装は知らない。

observe で「映像→特徴量→観測」を作り、classify で判定する。両者を分けてあるのは、
キャリブレーション中は観測だけ作って基準採取に回し、判定はしないため。
"""

from __future__ import annotations

from .contracts import Assessment, CalibrationProfile, Frame, Observation
from .features.normalize import normalize_features
from .features.rppg import RppgEstimator
from .ports import Classifier, FeatureExtractor, LandmarkDetector
from .temporal import TemporalContext


class Pipeline:
    def __init__(
        self,
        detector: LandmarkDetector,
        extractor: FeatureExtractor,
        classifier: Classifier,
        temporal: TemporalContext,
        normalize_version: int = 1,
        profile: CalibrationProfile | None = None,
        rppg: RppgEstimator | None = None,
    ) -> None:
        self._detector = detector
        self._extractor = extractor
        self._classifier = classifier
        self._temporal = temporal
        self._version = normalize_version
        self._profile = profile or CalibrationProfile.identity()
        self._rppg = rppg

    def set_profile(self, profile: CalibrationProfile) -> None:
        self._profile = profile

    def reset_state(self) -> None:
        """溜め込んだ状態を捨てる。再キャリブレーションの直前に呼ぶ。

        新しい基準で正規化した値と、古い基準で正規化した履歴は混ぜられない。
        判定側が持つ安静基準・注意残高も、別人には当てはまらない。
        """
        self._temporal.reset()
        if self._rppg is not None:
            self._rppg.reset()
        reset = getattr(self._classifier, "reset", None)
        if callable(reset):
            reset()

    def observe(self, frame: Frame) -> Observation:
        landmarks = self._detector.detect(frame)
        raw = self._extractor.extract(landmarks, frame.timestamp)
        features = normalize_features(raw, self._profile, self._version)
        if self._rppg is not None and features.face_present:
            # 心拍は複数フレームから出るので、正規化後の袋へ足す（キャリブ不要の量）。
            features = self._rppg.augment(frame, landmarks, features)
        self._temporal.append(features)
        return Observation(
            frame=frame,
            landmarks=landmarks,
            features=features,
            history=self._temporal,
            profile=self._profile,
        )

    def classify(self, obs: Observation) -> Assessment:
        return self._classifier.assess(obs)

    def close(self) -> None:
        self._detector.close()
