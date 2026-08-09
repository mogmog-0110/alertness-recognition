"""PSG 関連APIの公開窓口と、配列・単一信号向けの簡易解析を提供する。

DROZY変換で使う正式なEDF読込、チャンネル解決、窓特徴抽出、およびデータ型は
``psg_recording`` から再公開し、呼び出し側が ``alertness.bio.psg`` だけに依存できるようにする。
一方、``read_psg_signal`` と ``build_psg_feature_series`` は単一チャンネルや既に読み込まれた
EEG/EOG配列を扱う軽量な入口で、簡易テキスト入力や互換用途・小規模テストを支える。

manifest生成の本経路は複数の指定EEG/EOGチャンネルを検証する ``read_psg`` と
``extract_psg_features`` を使う。簡易関数は同名の特徴キーを返すが、周波数ビンを概算する補助API
であり、正式なDROZY信号処理や教師ラベル生成の代替ではない。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .psg_recording import (
    DEFAULT_CHANNEL_ALIASES,
    PsgFeature,
    PsgRecording,
    extract_psg_features,
    read_psg,
    resolve_channels,
)

__all__ = [
    "DEFAULT_CHANNEL_ALIASES",
    "PsgFeature",
    "PsgRecording",
    "build_psg_feature_series",
    "extract_psg_features",
    "read_psg",
    "read_psg_signal",
    "resolve_channels",
]


def read_psg_signal(path: str | Path, *, channel: str | None = None) -> tuple[np.ndarray, int]:
    """EDF もしくは簡易テキストから信号を読み込む。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PSG file not found: {p}")

    if p.suffix.lower() == ".edf":
        try:
            import pyedflib
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "pyedflib が必要です。pip install pyedflib で導入してください。"
            ) from exc
        with pyedflib.EdfReader(str(p)) as reader:
            if channel is None:
                channel = reader.getSignalLabels()[0]
            channel_index = reader.getSignalLabels().index(channel)
            signal = reader.readSignal(channel_index)
            sample_rate = int(reader.getSampleFrequencies()[channel_index])
            return np.asarray(signal, dtype=float), sample_rate

    with p.open("r", encoding="utf-8") as handle:
        values = [float(line.strip()) for line in handle if line.strip()]
    return np.asarray(values, dtype=float), 1


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
        theta_values = spectrum[np.min([2, len(spectrum) - 1]) : np.min([3, len(spectrum)])]
        alpha_values = spectrum[np.min([4, len(spectrum) - 1]) : np.min([5, len(spectrum)])]
        theta = float(np.mean(theta_values))
        alpha = float(np.mean(alpha_values))
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
