"""拍の検出。PPG/ECG などの波形から、1拍ごとのピーク位置を返す。

rPPG の脈波は接触センサと違い、呼吸による基線の揺れと照明由来の雑音が同じ大きさで
乗っている。素朴な「全体を標準偏差で正規化して固定しきい値」だと次の壊れ方をする:

- 基線が揺れている間、しきい値が実質的に上下し、山の小さい区間の拍を丸ごと落とす。
  拍を1つ落とすと、その RR 間隔だけが 2 倍になる。RMSSD は隣接 RR の差の二乗平均なので、
  1回の取りこぼしが値を跳ね上げる。実収録で RMSSD の中央値が 225ms（拍間隔の 24%、
  人の安静時 20〜50ms とは桁違い）だったのはこれが原因。
- 雑音の突起が本物の拍より高いと、貪欲に高い順で採る方式ではその突起が先に採られ、
  不応期に入る本物の拍が捨てられる。

そこで3段構えにする:
1. 心拍の帯域だけ残す（呼吸の揺れと高域の雑音を先に落とす）
2. しきい値を局所化する（区間ごとの実効値に対する比で見る）
3. 時間順に走査し、不応期内で競合したら高い方を残す（並びを壊さない）

それでも取りこぼしはゼロにならないので、拍間隔の側でも異常値を弾く（hrv.plausible_rr）。
HRV の解析では、指標を出す前にアーティファクト補正を入れるのが標準的な作法。
"""

from __future__ import annotations

import numpy as np


def upsample_bandlimited(signal: np.ndarray, factor: int) -> np.ndarray:
    """帯域制限補間で標本数を factor 倍にする（FFT のゼロ詰め）。

    拍の時刻はフレーム間隔に量子化される。30fps なら 33ms 刻みで、これは人の RMSSD
    （安静時 20〜50ms）と同じ桁なので、そのまま拍間隔を作ると心臓ではなくフレーム格子を
    測ることになる。脈波は帯域制限されているので、標本の間を理論通り埋められる。
    """
    x = np.asarray(signal, dtype=float).ravel()
    if factor <= 1 or x.size < 4:
        return x
    return np.fft.irfft(np.fft.rfft(x), x.size * factor) * factor


def bandpass(signal: np.ndarray, fs: float, min_bpm: float, max_bpm: float) -> np.ndarray:
    """心拍の帯域だけ残す（FFT で帯域外のビンを落とす）。

    呼吸による基線の揺れ（0.1〜0.5Hz）は脈波と同じかそれ以上の振幅で乗る。落とさずに
    山を探すと、基線が下がっている区間の拍がしきい値に届かず丸ごと消える。
    上端を切るのは、照明のちらつきなど拍より速い突起を拍と間違えないため。

    帯域は上下に少し広げてある。拍の波形は正弦ではないので、基本波の帯域ちょうどで
    切ると山が鈍り、ピーク位置が本来よりずれる。
    """
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 8 or fs <= 0:
        return x
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    spectrum = np.fft.rfft(x - np.mean(x))
    low = min_bpm / 60.0 * 0.8
    high = max_bpm / 60.0 * 1.5
    return np.fft.irfft(spectrum * _band_gain(freqs, low, high), x.size)


def _band_gain(freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    """帯域の重み。縁は余弦でなだらかに落とす。

    矩形で切ると時間領域にリンギング（ギブス現象）が出て、窓の端にある拍の位置が
    数十 ms ずれる。RMSSD は隣接差を見るので、端の1拍のずれがそのまま値に乗る。
    遷移帯は通過帯の端から 1/4 オクターブぶん取る。
    """
    gain = np.ones_like(freqs)
    lo_edge, hi_edge = low * 0.75, high * 1.25
    rising = (freqs > lo_edge) & (freqs < low)
    falling = (freqs > high) & (freqs < hi_edge)
    gain[freqs <= lo_edge] = 0.0
    gain[freqs >= hi_edge] = 0.0
    gain[rising] = 0.5 * (1.0 - np.cos(np.pi * (freqs[rising] - lo_edge) / (low - lo_edge)))
    gain[falling] = 0.5 * (1.0 + np.cos(np.pi * (freqs[falling] - high) / (hi_edge - high)))
    return gain


def peak_times(
    signal: np.ndarray,
    fs: float,
    min_bpm: float = 40.0,
    max_bpm: float = 180.0,
    upsample: int = 16,
) -> np.ndarray:
    """波形から拍の時刻[秒]を返す。帯域を絞ってから、標本の間を補間して探す。

    RMSSD のような拍間隔の指標は、ピーク位置の誤差がそのまま効くので、こちらを使う。
    """
    if fs <= 0:
        return np.empty(0, dtype=float)
    clean = bandpass(signal, fs, min_bpm, max_bpm)
    factor = max(1, upsample)
    dense = upsample_bandlimited(clean, factor)
    peaks = detect_peaks(dense, fs * factor, min_bpm, max_bpm) / (fs * factor)
    # 端の拍は落とす。周波数側の処理（帯域の絞り込みと帯域制限補間）はどちらも窓が
    # 周期的だと仮定するので、その仮定が最も崩れる窓の両端では山の位置がずれる。
    # 20 秒窓なら 20 拍前後あるので、2 拍削っても指標の精度には響かない。
    return peaks[1:-1] if peaks.size >= 4 else peaks


# 局所実効値の何倍を山とみなすか。小さいほど拾いすぎ、大きいほど取りこぼす。
_THRESHOLD_K = 0.6


def detect_peaks(
    signal: np.ndarray,
    fs: float,
    min_bpm: float = 40.0,
    max_bpm: float = 180.0,
) -> np.ndarray:
    """波形から拍のピークのサンプル位置(index)を返す。

    signal: 1次元の波形（PPG など）。fs: サンプリング周波数[Hz]。
    min_bpm/max_bpm: 想定する心拍の範囲。max_bpm から拍間の最小サンプル数（不応期）を、
    min_bpm から局所しきい値を測る窓幅を決める。
    """
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 3 or fs <= 0:
        return np.empty(0, dtype=int)

    x = x - np.mean(x)
    if np.std(x) < 1e-12:
        return np.empty(0, dtype=int)

    min_gap = max(1, int(round(fs * 60.0 / max_bpm)))
    # しきい値は「最も遅い心拍で2拍ぶん」の窓で測る。1拍より短いと山そのものが窓の
    # 実効値を押し上げ、自分のしきい値を自分で上げてしまう。
    window = max(3, int(round(fs * 60.0 / min_bpm * 2.0)))
    floor = _THRESHOLD_K * _moving_rms(x, window)

    rising = (x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])
    tall = x[1:-1] >= floor[1:-1]
    candidates = np.flatnonzero(rising & tall) + 1
    if candidates.size == 0:
        return np.empty(0, dtype=int)
    return _enforce_refractory(x, candidates, min_gap)


def _moving_rms(x: np.ndarray, window: int) -> np.ndarray:
    """窓ごとの実効値。端は窓が短くなるぶんだけ実際の標本数で割る。"""
    padded = np.concatenate(([0.0], np.cumsum(x * x)))
    half = window // 2
    lo = np.clip(np.arange(x.size) - half, 0, x.size)
    hi = np.clip(np.arange(x.size) + half + 1, 0, x.size)
    counts = np.maximum(1, hi - lo)
    return np.sqrt((padded[hi] - padded[lo]) / counts)


def _enforce_refractory(x: np.ndarray, candidates: np.ndarray, min_gap: int) -> np.ndarray:
    """不応期より近い山どうしは高い方だけ残す。時間順に走査して並びを保つ。

    高い順に貪欲に採ると、雑音の突起が本物の拍より高い場合にその突起が先に確定し、
    近くの本物の拍が不応期で捨てられる。時間順なら、競合は必ず隣どうしの比較になる。
    """
    chosen: list[int] = []
    for i in candidates:
        if chosen and i - chosen[-1] < min_gap:
            if x[i] > x[chosen[-1]]:
                chosen[-1] = int(i)
            continue
        chosen.append(int(i))
    return np.array(chosen, dtype=int)
