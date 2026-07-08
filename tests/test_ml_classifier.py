from __future__ import annotations

import pytest
from _helpers import make_observation

from alertness.classifier.ml_based import MLClassifier
from alertness.contracts import Features, Level


class _StubModel:
    """sklearn 互換の最小スタブ。predict に渡された X を記録する。"""

    def __init__(self, predicted: str, classes=None, proba=None) -> None:
        self._predicted = predicted
        self.classes_ = classes
        self._proba = proba
        self.seen_x: list | None = None

    def predict(self, x):
        self.seen_x = x
        return [self._predicted]

    def predict_proba(self, x):
        return [self._proba]


class _NoProbaModel:
    """predict_proba を持たないモデル（段階から score を出す経路の確認用）。"""

    def __init__(self, predicted: str) -> None:
        self._predicted = predicted

    def predict(self, x):
        return [self._predicted]


def _obs(values: dict, timestamp: float = 1.5):
    return make_observation(Features(values=values, timestamp=timestamp))


def _bundle(models: dict, features):
    return {"models": models, "features": list(features)}


def test_axes_are_mapped_and_levels_predicted():
    clf = MLClassifier(
        _bundle(
            {
                "label_drowsiness": _StubModel("high"),
                "label_distraction": _StubModel("none"),
            },
            ["ear", "gaze_off"],
        )
    )
    a = clf.assess(_obs({"ear": 0.1, "gaze_off": 0.0}))

    assert set(a.dimensions) == {"drowsiness", "distraction"}
    assert a.dimensions["drowsiness"].level is Level.HIGH
    assert a.dimensions["distraction"].level is Level.NONE
    assert a.timestamp == 1.5


def test_feature_vector_follows_bundle_order():
    model = _StubModel("none")
    clf = MLClassifier(_bundle({"label_drowsiness": model}, ["gaze_off", "ear"]))
    clf.assess(_obs({"ear": 0.3, "gaze_off": 0.9}))

    # bundle の順（gaze_off, ear）で並ぶこと。CSV の列順と一致する契約。
    assert model.seen_x == [[0.9, 0.3]]


def test_missing_feature_defaults_to_zero():
    model = _StubModel("none")
    clf = MLClassifier(_bundle({"label_drowsiness": model}, ["ear", "jawOpen"]))
    clf.assess(_obs({"ear": 0.25}))

    assert model.seen_x == [[0.25, 0.0]]


def test_severity_uses_probability_expectation():
    classes = ["high", "low", "medium", "none"]  # 並びが順序と違っても正しく重み付く
    proba = [0.5, 0.0, 0.0, 0.5]  # high と none が半々 → 期待段階 1.5/3
    clf = MLClassifier(
        _bundle({"label_drowsiness": _StubModel("high", classes, proba)}, ["ear"])
    )
    score = clf.assess(_obs({"ear": 0.1})).dimensions["drowsiness"].score

    assert score == pytest.approx(0.5)


def test_severity_falls_back_to_level_without_proba():
    clf = MLClassifier(_bundle({"label_drowsiness": _NoProbaModel("medium")}, ["ear"]))
    score = clf.assess(_obs({"ear": 0.1})).dimensions["drowsiness"].score

    assert score == pytest.approx(2 / 3)  # medium=2, high=3


def test_empty_models_rejected():
    with pytest.raises(ValueError, match="モデル"):
        MLClassifier(_bundle({}, ["ear"]))


def test_empty_features_rejected():
    with pytest.raises(ValueError, match="列順"):
        MLClassifier(_bundle({"label_drowsiness": _StubModel("none")}, []))


def test_unknown_level_rejected():
    clf = MLClassifier(_bundle({"label_drowsiness": _NoProbaModel("sleepy")}, ["ear"]))
    with pytest.raises(ValueError, match="未知のレベル"):
        clf.assess(_obs({"ear": 0.1}))
