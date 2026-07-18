from __future__ import annotations

import math

import numpy as np
import pytest

from alertness.bio import (
    detect_peaks,
    mean_hr,
    pnn50,
    rmssd,
    rr_intervals_ms,
    sdnn,
    stage_from_rmssd,
)


def _render_ppg(beat_times_s, seconds, fs, width=0.05):
    t = np.arange(0, seconds, 1.0 / fs)
    signal = np.zeros_like(t)
    for bt in beat_times_s:
        signal += np.exp(-0.5 * ((t - bt) / width) ** 2)
    return signal


def test_detect_peaks_recovers_regular_beats():
    fs = 64.0
    beats = np.arange(1.0, 10.0, 60.0 / 72.0)  # 72bpm の等間隔拍
    ppg = _render_ppg(beats, 10.0, fs)
    peaks = detect_peaks(ppg, fs)
    # 拍数が概ね一致し、復元した心拍が 72bpm 近傍。
    assert abs(len(peaks) - len(beats)) <= 1
    hr = mean_hr(rr_intervals_ms(peaks / fs))
    assert abs(hr - 72.0) < 5.0


def test_detect_peaks_empty_on_flat_signal():
    assert detect_peaks(np.zeros(200), 64.0).size == 0


def test_rr_intervals_needs_two_beats():
    assert rr_intervals_ms([1.0]).size == 0
    rr = rr_intervals_ms([1.0, 2.0, 3.2])
    assert np.allclose(rr, [1000.0, 1200.0])


def test_rmssd_larger_for_more_variable_rr():
    steady = rr_intervals_ms(np.cumsum([0, 0.83, 0.83, 0.83, 0.83]))
    jittery = rr_intervals_ms(np.cumsum([0, 0.7, 0.95, 0.72, 0.98]))
    assert rmssd(jittery) > rmssd(steady)


def test_hrv_metrics_nan_when_insufficient():
    empty = rr_intervals_ms([1.0])
    assert math.isnan(sdnn(empty))
    assert math.isnan(rmssd(empty))
    assert math.isnan(pnn50(empty))
    assert math.isnan(mean_hr(empty))


def test_stage_from_rmssd_is_inverse():
    thresholds = (50.0, 35.0, 20.0)
    assert stage_from_rmssd(60.0, thresholds) == "none"
    assert stage_from_rmssd(40.0, thresholds) == "low"
    assert stage_from_rmssd(25.0, thresholds) == "medium"
    assert stage_from_rmssd(10.0, thresholds) == "high"


def test_stage_from_rmssd_unknown_is_empty():
    # 拍が足りず計算不能(nan)なら段階を断定しない。
    assert stage_from_rmssd(float("nan"), (50.0, 35.0, 20.0)) == ""


def test_stage_from_rmssd_rejects_ascending_thresholds():
    with pytest.raises(ValueError):
        stage_from_rmssd(30.0, (20.0, 35.0, 50.0))
