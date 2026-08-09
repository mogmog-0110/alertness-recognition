from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _signal_module():
    try:
        from scipy import signal
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            'DROZY の信号処理には scipy が必要です。pip install -e ".[drozy]" を実行してください。'
        ) from exc
    return signal


def bandpass(values: np.ndarray, sample_rate: float, low_hz: float, high_hz: float) -> np.ndarray:
    """ゼロ位相Butterworthフィルタ。短い入力では因果フィルタへフォールバックする。"""
    if sample_rate <= 0:
        raise ValueError("sample_rate は正の値である必要があります")
    nyquist = sample_rate / 2.0
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(f"不正な帯域です: {low_hz}-{high_hz} Hz (fs={sample_rate})")
    signal = _signal_module()
    sos = signal.butter(4, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
    arr = np.asarray(values, dtype=float)
    if arr.size < 16:
        return signal.sosfilt(sos, arr)
    try:
        return signal.sosfiltfilt(sos, arr)
    except ValueError:
        return signal.sosfilt(sos, arr)


def relative_band_powers(
    values: np.ndarray,
    sample_rate: float,
    bands: Mapping[str, tuple[float, float]],
    *,
    total_band: tuple[float, float] = (0.5, 35.0),
) -> dict[str, float]:
    """Welch PSDから指定帯域の相対パワーを返す。"""
    signal = _signal_module()
    arr = np.asarray(values, dtype=float)
    if arr.size < 4 or not np.all(np.isfinite(arr)) or float(np.std(arr)) <= 1e-12:
        return {name: float("nan") for name in bands}
    nperseg = min(arr.size, max(8, int(round(sample_rate * 2.0))))
    frequencies, psd = signal.welch(arr, fs=sample_rate, nperseg=nperseg)
    total_mask = (frequencies >= total_band[0]) & (frequencies < total_band[1])
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    total = float(integrate(psd[total_mask], frequencies[total_mask]))
    if total <= 1e-15:
        return {name: float("nan") for name in bands}
    output: dict[str, float] = {}
    for name, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies < high)
        output[name] = float(integrate(psd[mask], frequencies[mask]) / total)
    return output
