"""拍の検出。PPG/ECG などの波形から、1拍ごとのピーク位置を返す。

素朴な「全体を std で正規化して固定しきい値で山を選ぶ」実装は、実データで破綻する。
UBFC-Phys の接触PPG(手首装着 E4)で確かめたところ、発話課題の区間で

  - 支配周波数が 0.77Hz(=46bpm)。心拍ではなく発話・体動によるベースライン変動
  - 窓ごとの振幅が std 109→23 と4.7倍に振れる
  - 結果、窓RMSSD が 300〜900ms（人の安静時 20〜50ms と桁が2つ違う）
  - 推定心拍が安静区間より低く出る（拍の取りこぼし）

という壊れ方をした。原因は2つで、(1) ベースライン変動を残したまま山を選ぶと、拍より
振幅の大きい変動の峰に引きずられる、(2) 全体の std で正規化すると、振幅の大きい区間に
スケールを引かれて静かな区間の拍がしきい値を下回る。

そこで3段構えにする。numpy のみで完結させるのは従来どおり。

  1. 帯域通過   … 心拍帯の外（ベースライン変動と高周波雑音）を落とす
  2. 局所正規化 … 数拍ぶんの移動RMSで割り、区間ごとの振幅差を吸収する
  3. 不応期     … 支配周波数から推定した拍間隔を基準にし、近すぎる山を捨てる

返り値の契約は従来どおり「拍のサンプル位置」なので、さらに頑健にしたくなったら
この関数だけを差し替えればよい。
"""

from __future__ import annotations

import numpy as np

# 局所振幅を測る移動窓の長さ。推定した拍間隔の何倍か。
# 短いと拍そのものを平滑化してしまい、長いと区間ごとの振幅差を吸収できない。
_LOCAL_WINDOW_BEATS = 3.0
# 局所振幅で正規化したあとの高さのしきい値。正規化後の脈波は RMS が 1 前後になるので、
# 山の頂点は 1.4 付近に来る。0.5 は「雑音ではないが、形が崩れた拍も拾う」あたり。
_HEIGHT_THRESHOLD = 0.5
# 不応期。推定した拍間隔に対する割合。心臓が拍間隔の半分で再び打つことはない。
# 固定の 60/max_bpm だと、実際の心拍が max_bpm よりずっと遅いときに不応期が短すぎて、
# 1拍の中の重複拍波(dicrotic notch)まで拾ってしまう。
_REFRACTORY_RATIO = 0.55
# 不応期の上限を決める心拍[bpm]。「これより速い拍は絶対に潰さない」という下限保証。
# 支配周波数は体動が乗ると心拍より遅い側へ引かれることがあり（UBFC-Phys の発話課題では
# 46bpm を指した）、その間隔から不応期を作ると本物の拍まで抑え込んでしまう。実測では
# この上限を入れると、汚れた区間の推定心拍が 40→50bpm へ戻り、きれいな区間は変化しなかった。
_REFRACTORY_MAX_BPM = 120.0
# 局所振幅の下限。無音区間で 0 に近いスケールで割ると雑音が増幅されるのを防ぐ。
_SCALE_FLOOR_RATIO = 0.3
# 帯域通過の前に反射パディングする長さ（信号長に対する割合）。
# FFT は信号を周期的とみなすので、窓の端と端が繋がらないと不連続になり、リンギングが
# 端の拍位置をずらす（合成波での実測で、先頭/末尾の RR だけ 30〜70ms 外れた）。
_PAD_RATIO = 0.25
# 遷移帯の幅（通過帯域幅に対する割合）。通過域の「外側」に置くので、通過域の利得は 1 のまま。
# 矩形の打ち切りは時間領域で sinc になり長く尾を引くため、余弦で滑らかに落とす。
# 広げすぎると下側の遷移が 0Hz に届いてベースライン変動を通してしまうので、狭めに取る。
_ROLLOFF_RATIO = 0.2


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


def peak_times(
    signal: np.ndarray,
    fs: float,
    min_bpm: float = 40.0,
    max_bpm: float = 180.0,
    upsample: int = 16,
) -> np.ndarray:
    """波形から拍の時刻[秒]を返す。標本の間は帯域制限補間で埋める。

    RMSSD のような拍間隔の指標は、ピーク位置の誤差がそのまま効くので、こちらを使う。
    """
    if fs <= 0:
        return np.empty(0, dtype=float)
    factor = max(1, upsample)
    dense = upsample_bandlimited(signal, factor)
    return detect_peaks(dense, fs * factor, min_bpm, max_bpm) / (fs * factor)


def _passband_gain(freqs: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    """通過域は利得1、その外側へ余弦で滑らかに落とす利得曲線。"""
    width = _ROLLOFF_RATIO * max(high_hz - low_hz, 1e-12)
    ramp_up = np.clip((freqs - (low_hz - width)) / width, 0.0, 1.0)
    ramp_down = np.clip(((high_hz + width) - freqs) / width, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(ramp_up * ramp_down, 0.0, 1.0))


def bandpass(signal: np.ndarray, fs: float, min_bpm: float, max_bpm: float) -> np.ndarray:
    """心拍帯の外を落とす。FFT による帯域通過。

    呼吸や体動によるベースライン変動は拍より遅く、振幅は桁で大きいことがある。残したまま
    山を選ぶと、その峰が拍として採られる。逆に高周波側の雑音は偽の局所最大を作る。

    端は反射パディングしてから濾波し、あとで切り戻す。FFT は信号を周期的とみなすため、
    窓の端どうしが繋がらないと不連続が生じ、そのリンギングが端の拍位置をずらす。
    """
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 8 or fs <= 0:
        return x - np.mean(x) if x.size else x

    x = x - np.mean(x)
    pad = max(1, min(int(x.size * _PAD_RATIO), x.size - 2))
    padded = np.concatenate((x[pad:0:-1], x, x[-2 : -pad - 2 : -1]))

    freqs = np.fft.rfftfreq(padded.size, d=1.0 / fs)
    gain = _passband_gain(freqs, min_bpm / 60.0, max_bpm / 60.0)
    filtered = np.fft.irfft(np.fft.rfft(padded) * gain, padded.size)
    return filtered[pad : pad + x.size]


def dominant_interval(signal: np.ndarray, fs: float, min_bpm: float, max_bpm: float) -> float:
    """帯域内の支配周波数から拍間隔[サンプル]を推定する。取れなければ nan。

    周波数領域の推定は個々の山の形に左右されないので、波形が崩れていても拍の「おおよその
    間隔」は取れる。これを不応期と局所窓の基準に使うことで、しきい値を心拍によらず決められる。
    """
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 8 or fs <= 0:
        return float("nan")
    windowed = (x - np.mean(x)) * np.hanning(x.size)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    band = (freqs >= min_bpm / 60.0) & (freqs <= max_bpm / 60.0)
    if not np.any(band) or float(np.sum(power[band])) <= 0.0:
        return float("nan")
    peak = float(freqs[band][np.argmax(power[band])])
    return fs / peak if peak > 0 else float("nan")


def _moving_rms(x: np.ndarray, width: int) -> np.ndarray:
    """幅 width の移動RMS。累積和で O(N)。端は窓を縮めて扱う。"""
    if width < 2 or width >= x.size:
        return np.full(x.size, float(np.sqrt(np.mean(x * x))))
    cumulative = np.concatenate(([0.0], np.cumsum(x * x)))
    starts = np.clip(np.arange(x.size) - width // 2, 0, x.size)
    ends = np.clip(starts + width, 0, x.size)
    return np.sqrt((cumulative[ends] - cumulative[starts]) / np.maximum(ends - starts, 1))


def _local_normalize(x: np.ndarray, width: int) -> np.ndarray:
    """数拍ぶんの移動RMSで割り、区間ごとの振幅差を吸収する。"""
    scale = _moving_rms(x, width)
    positive = scale[scale > 0.0]
    if positive.size == 0:
        return x
    floor = _SCALE_FLOOR_RATIO * float(np.median(positive))
    return x / np.maximum(scale, max(floor, 1e-12))


def _pick(x: np.ndarray, min_gap: int, threshold: float) -> np.ndarray:
    """しきい値を超える局所最大を、高い順に不応期を守って採る。"""
    inner = x[1:-1]
    candidates = np.flatnonzero((inner > x[:-2]) & (inner >= x[2:]) & (inner >= threshold)) + 1
    if candidates.size == 0:
        return np.empty(0, dtype=int)
    chosen: list[int] = []
    for i in candidates[np.argsort(x[candidates])[::-1]]:
        if all(abs(int(i) - j) >= min_gap for j in chosen):
            chosen.append(int(i))
    return np.array(sorted(chosen), dtype=int)


def detect_peaks(
    signal: np.ndarray,
    fs: float,
    min_bpm: float = 40.0,
    max_bpm: float = 180.0,
) -> np.ndarray:
    """波形から拍のピークのサンプル位置(index)を返す。

    signal: 1次元の波形（PPG など）。fs: サンプリング周波数[Hz]。
    min_bpm/max_bpm: 想定する心拍の範囲。帯域通過の通過域になる。
    """
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 3 or fs <= 0:
        return np.empty(0, dtype=int)

    x = bandpass(x, fs, min_bpm, max_bpm)
    if float(np.std(x)) < 1e-12:
        return np.empty(0, dtype=int)

    # 拍間隔が取れないほど成分が無いときは、想定最大心拍から不応期を決める（従来と同じ挙動）。
    interval = dominant_interval(x, fs, min_bpm, max_bpm)
    if not np.isfinite(interval) or interval <= 0:
        interval = fs * 60.0 / max_bpm

    x = _local_normalize(x, int(round(_LOCAL_WINDOW_BEATS * interval)))
    # 推定間隔から不応期を作るが、速い拍を潰さないよう上限を掛ける。
    refractory = min(_REFRACTORY_RATIO * interval, fs * 60.0 / _REFRACTORY_MAX_BPM)
    return _pick(x, max(1, int(round(refractory))), _HEIGHT_THRESHOLD)
