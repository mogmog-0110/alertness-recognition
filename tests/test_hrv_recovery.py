"""雑音と基線の揺れが乗った脈波から HRV を取り戻せるかのテスト。

正解の分かる合成波で測る。実収録では真の RMSSD が分からないので、まずここで
「取れるはずの条件で取れる」ことを確かめてから実データに進む。
"""

from __future__ import annotations

import numpy as np

from alertness.bio import plausible_rr, rmssd, rr_intervals_ms
from alertness.bio.peaks import bandpass, detect_peaks, peak_times

FS = 30.0
SECONDS = 30.0


def _beat_times(mean_rr: float, jitter_ms: float, seconds: float, seed: int = 0):
    """指定の RMSSD になるよう、拍ごとに交互のゆらぎを与えた拍時刻。

    交互にずらすと隣接差が一定になり、RMSSD の正解値が jitter から直に決まる。
    """
    rng = np.random.default_rng(seed)
    times = [1.0]
    sign = 1.0
    while times[-1] < seconds - 1.0:
        rr = mean_rr + sign * jitter_ms / 2000.0 + rng.normal(0.0, 0.002)
        times.append(times[-1] + rr)
        sign = -sign
    return np.array(times[:-1])


def _render(beats, seconds=SECONDS, fs=FS, width=0.06, noise=0.0, wander=0.0, seed=1):
    """脈波を描く。noise で白色雑音、wander で呼吸による基線の揺れを足せる。"""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds, 1.0 / fs)
    signal = np.zeros_like(t)
    for bt in beats:
        signal += np.exp(-0.5 * ((t - bt) / width) ** 2)
    if wander:
        signal += wander * np.sin(2.0 * np.pi * 0.25 * t)  # 15回/分の呼吸
    if noise:
        signal += rng.normal(0.0, noise, t.size)
    return signal


def _recovered_rmssd(signal):
    times = peak_times(signal, FS, 42.0, 180.0, upsample=16)
    rr = rr_intervals_ms(times)
    if rr.size < 4:
        return float("nan")
    return rmssd(rr, plausible_rr(rr))


def test_a_clean_pulse_gives_the_expected_rmssd():
    beats = _beat_times(60.0 / 70.0, jitter_ms=40.0, seconds=SECONDS)
    value = _recovered_rmssd(_render(beats))
    assert abs(value - 40.0) < 15.0


def test_baseline_wander_no_longer_swallows_beats():
    # 呼吸の揺れは脈波と同じ大きさで乗る。帯域を絞らないと、基線が下がっている区間の
    # 拍がしきい値に届かず消え、その RR だけが 2 倍になって RMSSD が跳ねる。
    beats = _beat_times(60.0 / 70.0, jitter_ms=40.0, seconds=SECONDS)
    signal = _render(beats, wander=1.5)
    assert abs(_recovered_rmssd(signal) - 40.0) < 20.0


def test_bandpass_removes_the_respiratory_swing():
    t = np.arange(0.0, 20.0, 1.0 / FS)
    respiration = np.sin(2.0 * np.pi * 0.25 * t)
    pulse = 0.3 * np.sin(2.0 * np.pi * 1.2 * t)
    filtered = bandpass(respiration + pulse, FS, 42.0, 180.0)
    # 呼吸成分がほぼ消え、脈の振幅は残る。
    assert float(np.std(filtered)) < float(np.std(respiration + pulse))
    assert float(np.std(filtered)) > 0.15


def test_a_noise_spike_does_not_displace_a_real_beat():
    # 高い順に貪欲に採ると、雑音の突起が先に確定して近くの本物の拍が捨てられる。
    beats = _beat_times(60.0 / 70.0, jitter_ms=30.0, seconds=SECONDS)
    signal = _render(beats)
    spike = int(FS * 5.0) + 3
    signal[spike] += 3.0  # 本物の拍より高い単発の突起
    peaks = detect_peaks(signal, FS, 42.0, 180.0)
    assert abs(len(peaks) - len(beats)) <= 2


def test_a_dropped_beat_is_rejected_instead_of_inflating_rmssd():
    # 取りこぼしを1つ混ぜる。除かなければ RMSSD は数百 ms に跳ねる（実収録で見た壊れ方）。
    rr = np.array([850.0, 870.0, 860.0, 1720.0, 855.0, 865.0])  # 4番目が2拍ぶん
    raw = rmssd(rr)
    cleaned = rmssd(rr, plausible_rr(rr))
    assert raw > 400.0
    assert cleaned < 40.0


def test_plausible_rr_keeps_a_normal_series_intact():
    rr = np.array([850.0, 870.0, 860.0, 855.0, 865.0])
    assert bool(np.all(plausible_rr(rr)))


def test_plausible_rr_drops_impossible_intervals():
    rr = np.array([850.0, 120.0, 860.0, 4000.0, 855.0])
    valid = plausible_rr(rr)
    assert not valid[1]  # 短すぎる
    assert not valid[3]  # 長すぎる


def test_rmssd_is_nan_when_no_pair_survives():
    # 測れないことと、変動が無いことは別。0 を返すと「完全に規則的」と読める。
    rr = np.array([850.0, 4000.0, 850.0])
    assert np.isnan(rmssd(rr, np.array([True, False, False])))


def test_recovered_rmssd_tracks_a_change_in_variability():
    # 絶対値が多少ずれても、変動が減ったことを追えなければストレス判定に使えない。
    relaxed = _render(_beat_times(60.0 / 70.0, 60.0, SECONDS, seed=2), noise=0.05)
    stressed = _render(_beat_times(60.0 / 70.0, 15.0, SECONDS, seed=3), noise=0.05)
    assert _recovered_rmssd(relaxed) > _recovered_rmssd(stressed)
