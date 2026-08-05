from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def build_psg_feature_series(
    eeg: Sequence[float] | np.ndarray,
    eog: Sequence[float] | np.ndarray,
    *,
    sample_rate: int = 512,
    window_seconds: float = 1.0,
) -> list[dict[str, float]]:
    """簡易の PSG 解析。実データの EDF でも同じインターフェースで扱えるようにする。"""
    eeg_arr = np.asarray(eeg, dtype=float)
    eog_arr = np.asarray(eog, dtype=float)

    if eeg_arr.size == 0 or eog_arr.size == 0:
        return []

    window_size = max(1, min(len(eeg_arr), len(eog_arr), int(sample_rate * window_seconds)))
    features: list[dict[str, float]] = []
    total_length = min(len(eeg_arr), len(eog_arr))
    for start in range(0, total_length, window_size):
        eeg_window = eeg_arr[start : start + window_size]
        eog_window = eog_arr[start : start + window_size]
        if eeg_window.size == 0 or eog_window.size == 0:
            continue

        spectrum = np.abs(np.fft.rfft(eeg_window))
        theta = float(np.mean(spectrum[np.min([2, len(spectrum) - 1]) : np.min([3, len(spectrum)])]))
        alpha = float(np.mean(spectrum[np.min([4, len(spectrum) - 1]) : np.min([5, len(spectrum)])]))
        beta = float(np.mean(spectrum[np.min([6, len(spectrum) - 1]) : np.min([7, len(spectrum)])]))
        sem = float(np.std(eog_window))
        blink_duration = float(np.mean(np.abs(eog_window)))
        microsleep_duration = float(np.sum(np.abs(eog_window) < 0.05)) / max(1, len(eog_window))

        features.append(
            {
                "theta": theta,
                "alpha": alpha,
                "beta": beta,
                "di": (theta + alpha) / max(beta, 1e-6),
                "sem": sem,
                "blink_duration": blink_duration,
                "microsleep_duration": microsleep_duration,
            }
        )

    return features
