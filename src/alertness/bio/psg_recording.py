"""DROZYのEDFを正準チャンネルへ読み込み、窓単位のPSG眠気特徴へ変換する。

配布EDFの表記揺れをエイリアスで解決し、5つのEEG（Fz/Cz/C3/C4/Pz）と水平・垂直EOGが
揃っていることを検証して ``PsgRecording`` に格納する。各チャンネルのサンプリング周波数を
保持するため、異なるEDF記録条件でも秒単位の共通窓として切り出せる。

``extract_psg_features`` は既定10秒窓を1秒ずつ進め、EEG各点の相対theta/alpha/betaを計算して
チャンネル中央値へ集約し、DIとEOG由来のSEM・瞬目・長時間閉眼相当時間を加える。窓中央時刻と
品質フラグを持つ ``PsgFeature`` は、変換器で動画timestampsと同期され、被験者内基準化、CDS、
LoD区間化へ順に渡される。

必須チャンネルの欠落や水平・垂直EOGの周波数不一致は推測で補完せずエラーにする。チャンネル
別の帯域値も保持し、集約結果の追跡や信号処理の検証に利用できるようにしている。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .eeg import bandpass, relative_band_powers
from .eog import extract_eog_features

DEFAULT_CHANNEL_ALIASES: dict[str, tuple[str, ...]] = {
    "fz": ("Fz", "EEG Fz-A1", "FZ-A1"),
    "cz": ("Cz", "EEG Cz-A1", "CZ-A1"),
    "c3": ("C3", "EEG C3-A1", "C3-A1"),
    "c4": ("C4", "EEG C4-A1", "C4-A1"),
    "pz": ("Pz", "EEG Pz-A1", "PZ-A1"),
    "horizontal_eog": ("Horizontal EOG", "HEOG", "EOG-H", "EOG H"),
    "vertical_eog": ("Vertical EOG", "VEOG", "EOG-V", "EOG V"),
}
EEG_CHANNELS = ("fz", "cz", "c3", "c4", "pz")


@dataclass(frozen=True)
class PsgRecording:
    signals: Mapping[str, np.ndarray]
    sample_rates: Mapping[str, float]
    source_labels: Mapping[str, str]

    @property
    def duration_seconds(self) -> float:
        durations = [
            len(values) / self.sample_rates[name]
            for name, values in self.signals.items()
            if self.sample_rates[name] > 0
        ]
        return min(durations, default=0.0)


@dataclass(frozen=True)
class PsgFeature:
    timestamp: float
    theta: float
    alpha: float
    beta: float
    di: float
    sem: float
    blink_duration: float
    microsleep_duration: float
    valid: bool
    eeg_channels: Mapping[str, Mapping[str, float]]

    def values(self) -> dict[str, float]:
        return {
            "theta": self.theta,
            "alpha": self.alpha,
            "beta": self.beta,
            "di": self.di,
            "sem": self.sem,
            "blink_duration": self.blink_duration,
            "microsleep_duration": self.microsleep_duration,
        }


def _normalized_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def resolve_channels(labels: Sequence[str], aliases: Mapping[str, Sequence[str]]) -> dict[str, int]:
    normalized = {_normalized_label(label): index for index, label in enumerate(labels)}
    resolved: dict[str, int] = {}
    for canonical, candidates in aliases.items():
        for candidate in (canonical, *candidates):
            index = normalized.get(_normalized_label(candidate))
            if index is not None:
                resolved[canonical] = index
                break
    required = {*EEG_CHANNELS, "horizontal_eog", "vertical_eog"}
    missing = sorted(required - resolved.keys())
    if missing:
        raise ValueError(f"EDF に必須チャンネルがありません: {', '.join(missing)}")
    return resolved


def read_psg(
    path: str | Path,
    *,
    channel_aliases: Mapping[str, Sequence[str]] | None = None,
) -> PsgRecording:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PSG file not found: {p}")
    if p.suffix.lower() != ".edf":
        raise ValueError(f"read_psg はEDFのみを受け付けます: {p}")
    try:
        import pyedflib
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "DROZY のEDF読込には pyedflib が必要です。"
            'pip install -e ".[drozy]" を実行してください。'
        ) from exc
    aliases = channel_aliases or DEFAULT_CHANNEL_ALIASES
    with pyedflib.EdfReader(str(p)) as reader:
        labels = [str(label).strip() for label in reader.getSignalLabels()]
        resolved = resolve_channels(labels, aliases)
        signals = {
            canonical: np.asarray(reader.readSignal(index), dtype=float)
            for canonical, index in resolved.items()
        }
        rates = {
            canonical: float(reader.getSampleFrequency(index))
            for canonical, index in resolved.items()
        }
        sources = {canonical: labels[index] for canonical, index in resolved.items()}
    return PsgRecording(signals=signals, sample_rates=rates, source_labels=sources)


def _window(values: np.ndarray, sample_rate: float, start: float, duration: float) -> np.ndarray:
    first = int(round(start * sample_rate))
    last = int(round((start + duration) * sample_rate))
    return np.asarray(values[first:last], dtype=float)


def extract_psg_features(
    recording: PsgRecording,
    *,
    window_seconds: float = 10.0,
    stride_seconds: float = 1.0,
    eeg_low_hz: float = 0.5,
    eeg_high_hz: float = 35.0,
    eog_low_hz: float = 0.1,
    eog_high_hz: float = 15.0,
    eog_event_z: float = 2.0,
    blink_min_seconds: float = 0.08,
    blink_max_seconds: float = 0.8,
    microsleep_min_seconds: float = 0.5,
) -> list[PsgFeature]:
    """10秒窓・1秒strideを既定としてPSG特徴量を生成する。"""
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("window_seconds と stride_seconds は正の値である必要があります")
    duration = recording.duration_seconds
    if duration + 1e-9 < window_seconds:
        return []
    bands = {"theta": (4.0, 8.0), "alpha": (8.0, 12.0), "beta": (12.0, 30.0)}
    output: list[PsgFeature] = []
    starts = np.arange(0.0, duration - window_seconds + 1e-9, stride_seconds)
    for start_value in starts:
        start = float(start_value)
        channel_features: dict[str, Mapping[str, float]] = {}
        for name in EEG_CHANNELS:
            rate = recording.sample_rates[name]
            raw = _window(recording.signals[name], rate, start, window_seconds)
            filtered = bandpass(raw, rate, eeg_low_hz, eeg_high_hz)
            channel_features[name] = relative_band_powers(filtered, rate, bands)
        aggregates = {
            band: float(np.nanmedian([values[band] for values in channel_features.values()]))
            for band in bands
        }
        h_rate = recording.sample_rates["horizontal_eog"]
        v_rate = recording.sample_rates["vertical_eog"]
        if abs(h_rate - v_rate) > 1e-6:
            raise ValueError("水平EOGと垂直EOGのサンプリング周波数が一致しません")
        horizontal = _window(recording.signals["horizontal_eog"], h_rate, start, window_seconds)
        vertical = _window(recording.signals["vertical_eog"], v_rate, start, window_seconds)
        eog = extract_eog_features(
            horizontal,
            vertical,
            h_rate,
            low_hz=eog_low_hz,
            high_hz=eog_high_hz,
            event_z=eog_event_z,
            blink_min_seconds=blink_min_seconds,
            blink_max_seconds=blink_max_seconds,
            microsleep_min_seconds=microsleep_min_seconds,
        )
        quality_values = [
            *aggregates.values(),
            eog.sem,
            eog.blink_duration,
            eog.microsleep_duration,
        ]
        beta = aggregates["beta"]
        di = (aggregates["theta"] + aggregates["alpha"]) / max(beta, 1e-12)
        output.append(
            PsgFeature(
                timestamp=start + window_seconds / 2.0,
                theta=aggregates["theta"],
                alpha=aggregates["alpha"],
                beta=beta,
                di=float(di),
                sem=eog.sem,
                blink_duration=eog.blink_duration,
                microsleep_duration=eog.microsleep_duration,
                valid=bool(np.all(np.isfinite(quality_values))) and bool(np.isfinite(di)),
                eeg_channels=channel_features,
            )
        )
    return output
