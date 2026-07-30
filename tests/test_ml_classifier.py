from __future__ import annotations

import pytest
from _helpers import make_observation

from alertness.classifier.ml_based import MLClassifier
from alertness.contracts import CalibrationProfile, Features, Level, Observation, Pose


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


def test_binary_labels_map_to_levels():
    # 2値で束ねて学習したモデル（calm/elevated）も Level に写せる。
    clf = MLClassifier(_bundle({"label_stress": _StubModel("elevated")}, ["ear"]))
    a = clf.assess(_obs({"ear": 0.1}))
    assert a.dimensions["stress"].level is Level.HIGH  # elevated → 警告あり

    clf = MLClassifier(_bundle({"label_stress": _StubModel("calm")}, ["ear"]))
    a = clf.assess(_obs({"ear": 0.1}))
    assert a.dimensions["stress"].level is Level.NONE  # calm → 警告なし


def test_coarse3_mid_maps_to_medium():
    clf = MLClassifier(_bundle({"label_stress": _StubModel("mid")}, ["ear"]))
    assert clf.assess(_obs({"ear": 0.1})).dimensions["stress"].level is Level.MEDIUM


def test_eda_label_source_maps_to_base_axis():
    # EDA で作ったラベル列 label_stress_eda も、アプリの評価軸 stress として出す。
    clf = MLClassifier(_bundle({"label_stress_eda": _StubModel("elevated")}, ["ear"]))
    a = clf.assess(_obs({"ear": 0.1}))
    assert set(a.dimensions) == {"stress"}
    assert a.dimensions["stress"].level is Level.HIGH


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


def test_rest_centering_subtracts_baseline_when_flagged():
    # rest_centered の bundle は、profile の安静基準を引いてからモデルに渡す。
    model = _StubModel("none")
    bundle = {
        "models": {"label_stress": model},
        "features": ["hr_bpm", "eyeSquintLeft"],
        "rest_centered": True,
    }
    clf = MLClassifier(bundle)
    profile = CalibrationProfile(
        ear_open_baseline=0.3,
        mar_neutral=0.0,
        head_pose_neutral=Pose(0.0, 0.0, 0.0),
        gaze_center=(0.5, 0.5),
        face_scale=1.0,
        baselines={"hr_bpm": 70.0, "eyeSquintLeft": 0.5},
    )
    obs = make_observation(Features(values={"hr_bpm": 82.0, "eyeSquintLeft": 0.6}, timestamp=1.0))
    obs = Observation(obs.frame, obs.landmarks, obs.features, obs.history, profile)
    clf.assess(obs)

    assert model.seen_x == [[12.0, pytest.approx(0.1)]]  # 82-70, 0.6-0.5


def test_no_centering_when_flag_absent():
    # rest_centered が無い bundle は、profile に基準があっても引かない（絶対値のまま）。
    model = _StubModel("none")
    clf = MLClassifier(_bundle({"label_stress": model}, ["hr_bpm"]))
    profile = CalibrationProfile(
        ear_open_baseline=0.3,
        mar_neutral=0.0,
        head_pose_neutral=Pose(0.0, 0.0, 0.0),
        gaze_center=(0.5, 0.5),
        face_scale=1.0,
        baselines={"hr_bpm": 70.0},
    )
    obs = make_observation(Features(values={"hr_bpm": 82.0}, timestamp=1.0))
    obs = Observation(obs.frame, obs.landmarks, obs.features, obs.history, profile)
    clf.assess(obs)

    assert model.seen_x == [[82.0]]


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
