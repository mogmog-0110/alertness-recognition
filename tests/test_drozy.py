import numpy as np

from alertness.bio.psg import build_psg_feature_series
from alertness.calibration.baseline import normalize_feature_series
from alertness.classifier.cds import compute_cds
from alertness.classifier.lod import classify_lod
from alertness.temporal import compress_segments, smooth_labels


def test_build_psg_feature_series_returns_sleep_related_features() -> None:
    sample_rate = 512
    t = np.arange(0, sample_rate, dtype=float)
    eeg = np.sin(2 * np.pi * 4 * t / sample_rate) + 0.2 * np.sin(2 * np.pi * 9 * t / sample_rate)
    eog = np.sin(2 * np.pi * 1 * t / sample_rate) + 0.8 * np.sin(2 * np.pi * 2 * t / sample_rate)

    features = build_psg_feature_series(eeg, eog, sample_rate=sample_rate, window_seconds=1.0)

    assert len(features) > 0
    assert "theta" in features[0]
    assert "alpha" in features[0]
    assert "sem" in features[0]
    assert "blink_duration" in features[0]


def test_normalize_feature_series_uses_baseline_statistics() -> None:
    features = [
        {"theta": 0.4, "alpha": 0.3, "beta": 0.6, "di": 0.8, "sem": 0.2, "blink_duration": 0.1},
        {"theta": 0.8, "alpha": 0.7, "beta": 0.5, "di": 1.5, "sem": 0.6, "blink_duration": 0.4},
    ]

    normalized = normalize_feature_series(features)

    assert len(normalized) == 2
    assert normalized[0]["theta"] < normalized[1]["theta"]
    assert normalized[0]["beta"] < normalized[1]["beta"]


def test_compute_cds_rises_for_sleepier_inputs() -> None:
    sleepy = [{"theta": 2.0, "alpha": 1.8, "beta": 0.2, "di": 2.0, "sem": 1.2, "blink_duration": 0.7, "microsleep_duration": 0.5}]
    alert = [{"theta": 0.3, "alpha": 0.2, "beta": 1.2, "di": 0.2, "sem": 0.1, "blink_duration": 0.1, "microsleep_duration": 0.0}]

    sleepy_score = compute_cds(sleepy)[0]
    alert_score = compute_cds(alert)[0]

    assert sleepy_score > alert_score


def test_classify_lod_maps_scores_to_levels() -> None:
    levels = classify_lod([5, 25, 60, 90])

    assert levels == ["none", "low", "medium", "high"]


def test_smooth_and_compress_labels_produce_segments() -> None:
    labels = ["none", "none", "low", "low", "low", "high", "high"]

    smoothed = smooth_labels(labels, window=3)
    segments = compress_segments(smoothed, min_duration=2)

    assert smoothed[2] == "low"
    assert len(segments) == 3
    assert segments[0]["label"] == "none"
    assert segments[1]["label"] == "low"
    assert segments[2]["label"] == "high"
