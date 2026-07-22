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


def test_peak_times_beats_frame_quantization():
    # ゆらぎゼロの脈。拍の時刻をフレーム格子のまま取ると RMSSD が 20ms 級で出てしまう
    # （人の安静時 20〜50ms と同じ桁）。帯域制限補間を入れると桁で下がる。
    from alertness.bio.peaks import peak_times

    fs = 30.0
    samples = np.arange(int(fs * 20)) / fs
    signal = np.sin(2 * np.pi * 61.0 / 60.0 * samples)

    raw = rmssd(rr_intervals_ms(peak_times(signal, fs, upsample=1)))
    fine = rmssd(rr_intervals_ms(peak_times(signal, fs, upsample=16)))
    assert raw > 15.0  # 補間なしでは量子化だけでこれだけ出る（測りたい範囲と同じ桁）
    assert fine < 10.0  # 補間すると効果（十数ms）より小さいところまで下がる
    assert fine < raw / 2.0


def test_upsample_bandlimited_preserves_shape_and_length():
    from alertness.bio.peaks import upsample_bandlimited

    fs = 30.0
    samples = np.arange(120) / fs
    signal = np.sin(2 * np.pi * 1.5 * samples)  # 4秒で6周期＝窓に対して周期的
    dense = upsample_bandlimited(signal, 4)
    assert dense.size == signal.size * 4
    assert np.allclose(dense[::4], signal, atol=1e-6)  # 元の標本はそのまま通る
    assert abs(float(np.max(np.abs(dense))) - 1.0) < 1e-3  # 振幅も保つ
