from __future__ import annotations

import pytest
from _helpers import FakeHistory, make_observation

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
    clf = MLClassifier(_bundle({"label_drowsiness": _StubModel("high", classes, proba)}, ["ear"]))
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


def _history_obs(ears, fps: float = 30.0):
    """ear が 0,1,2... と増える履歴を作り、その最後のフレームを現在の観測にする。"""
    frames = [Features(values={"ear": float(e)}, timestamp=i / fps) for i, e in enumerate(ears)]
    return make_observation(frames[-1], FakeHistory(frames, fps))


def test_window_defaults_to_one_frame():
    # window を持たない bundle（SVM/RandomForest 等）は従来どおりフレーム単位。
    model = _StubModel("none")
    clf = MLClassifier(_bundle({"label_drowsiness": model}, ["ear"]))
    clf.assess(_history_obs([1, 2, 3]))

    assert clf.window == 1
    assert model.seen_x == [[3.0]]  # 履歴があっても現在フレームだけ


def test_window_builds_sequence_from_history():
    # window>1（LSTM 等）は直近 window フレームを古い順に並べた行列を渡す。
    model = _StubModel("high")
    bundle = {"models": {"label_drowsiness": model}, "features": ["ear"], "window": 3}
    clf = MLClassifier(bundle)
    clf.assess(_history_obs([0, 1, 2, 3, 4]))

    assert clf.window == 3
    assert model.seen_x == [[[2.0], [3.0], [4.0]]]  # 末尾3フレーム、古い順


def test_window_pads_when_history_is_short():
    # 起動直後で履歴が足りないときは、最も古い行を複製して前に詰め、判定を止めない。
    model = _StubModel("low")
    bundle = {"models": {"label_drowsiness": model}, "features": ["ear"], "window": 4}
    clf = MLClassifier(bundle)
    clf.assess(_history_obs([7, 8]))

    assert model.seen_x == [[[7.0], [7.0], [7.0], [8.0]]]


def test_missing_features_become_present_flags():
    """rPPG のように欠ける特徴は、値だけでなく「あったか」も渡す。

    欠損を 0 で埋めるだけだと、心拍0bpm という実在しない値を学習側に渡すことになる。
    学習側(alertness-colab)が <列名>_present を作るので、推論も同じ規約で埋める。
    """
    import math

    from alertness.classifier.ml_based import MLClassifier

    seen: list[list[float]] = []

    class _Model:
        classes_ = ["none"]

        def predict(self, rows):
            seen.append(list(rows[0]))
            return ["none"]

    bundle = {
        "models": {"label_stress": _Model()},
        "features": ["ear_norm", "hr_bpm", "hr_bpm_present"],
    }
    classifier = MLClassifier(bundle)

    classifier.assess(_obs({"ear_norm": 1.0, "hr_bpm": 72.0}))
    assert seen[-1] == [1.0, 72.0, 1.0]  # 値があるので present=1

    classifier.assess(_obs({"ear_norm": 1.0, "hr_bpm": float("nan")}))
    assert seen[-1] == [1.0, 0.0, 0.0]  # NaN は 0 に埋めつつ present=0

    classifier.assess(_obs({"ear_norm": 1.0}))
    assert seen[-1] == [1.0, 0.0, 0.0]  # 列そのものが無くても同じ扱い
    assert not any(math.isnan(v) for row in seen for v in row)
