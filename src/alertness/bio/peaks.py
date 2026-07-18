"""拍の検出。PPG/ECG などの波形から、1拍ごとのピーク位置を返す。

接触センサの比較的きれいな波形を想定した素朴な検出（numpy のみ）。適応しきい値と
最小間隔（想定最大心拍から決まる不応期）で、山を貪欲に選ぶ。厳密さより「まず動く」を優先。
より頑健にしたくなったら、この関数だけを差し替えればよい（返り値の契約は拍のサンプル位置）。
"""

from __future__ import annotations

import numpy as np


def detect_peaks(
    signal: np.ndarray,
    fs: float,
    min_bpm: float = 40.0,
    max_bpm: float = 180.0,
) -> np.ndarray:
    """波形から拍のピークのサンプル位置(index)を返す。

    signal: 1次元の波形（PPG など）。fs: サンプリング周波数[Hz]。
    min_bpm/max_bpm: 想定する心拍の範囲。max_bpm から拍間の最小サンプル数（不応期）を決める。
    """
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 3 or fs <= 0:
        return np.empty(0, dtype=int)

    # 直流成分を抜き、振幅で正規化して、しきい値を波形スケールに依存させない。
    x = x - np.mean(x)
    std = np.std(x)
    if std < 1e-12:
        return np.empty(0, dtype=int)
    x = x / std

    min_gap = max(1, int(round(fs * 60.0 / max_bpm)))
    threshold = 0.3  # 正規化後の高さ。低すぎる山（ノイズ）を捨てる。

    # まず局所最大の候補を集める。
    candidates = [
        i
        for i in range(1, x.size - 1)
        if x[i] > x[i - 1] and x[i] >= x[i + 1] and x[i] >= threshold
    ]
    if not candidates:
        return np.empty(0, dtype=int)

    # 高い山から採り、既に採った拍と min_gap 未満で近すぎるものは捨てる（貪欲）。
    order = sorted(candidates, key=lambda i: x[i], reverse=True)
    chosen: list[int] = []
    for i in order:
        if all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(i)
    return np.array(sorted(chosen), dtype=int)
