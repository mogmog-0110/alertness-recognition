from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .eeg import bandpass


@dataclass(frozen=True)
class EogFeatures:
    sem: float
    blink_duration: float
    microsleep_duration: float


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def extract_eog_features(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    sample_rate: float,
    *,
    low_hz: float = 0.1,
    high_hz: float = 15.0,
    event_z: float = 2.0,
    blink_min_seconds: float = 0.08,
    blink_max_seconds: float = 0.8,
    microsleep_min_seconds: float = 0.5,
) -> EogFeatures:
    """EOG窓からSEM強度、平均瞬目時間、長時間閉眼相当時間を抽出する。"""
    h = np.asarray(horizontal, dtype=float)
    v = np.asarray(vertical, dtype=float)
    if h.size == 0 or v.size == 0 or not np.all(np.isfinite(h)) or not np.all(np.isfinite(v)):
        return EogFeatures(float("nan"), float("nan"), float("nan"))
    h_filtered = bandpass(h, sample_rate, low_hz, high_hz)
    v_filtered = bandpass(v, sample_rate, low_hz, high_hz)
    scale = 1.4826 * float(np.median(np.abs(v_filtered - np.median(v_filtered))))
    if scale <= 1e-12:
        return EogFeatures(0.0, 0.0, 0.0)
    vertical_z = np.abs((v_filtered - np.median(v_filtered)) / scale)
    event_runs = _runs(vertical_z >= event_z)
    durations = [(end - start) / sample_rate for start, end in event_runs]
    blinks = [d for d in durations if blink_min_seconds <= d <= blink_max_seconds]
    microsleeps = [d for d in durations if d >= microsleep_min_seconds]

    # SEMは水平EOGの低速な変動をロバスト尺度で正規化したRMSとして表す。
    h_scale = 1.4826 * float(np.median(np.abs(h_filtered - np.median(h_filtered))))
    sem = 0.0 if h_scale <= 1e-12 else float(np.sqrt(np.mean((h_filtered / h_scale) ** 2)))
    return EogFeatures(
        sem=sem,
        blink_duration=float(np.mean(blinks)) if blinks else 0.0,
        microsleep_duration=float(max(microsleeps)) if microsleeps else 0.0,
    )
