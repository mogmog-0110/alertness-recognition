"""生体信号(PPG)からストレスの教師ラベルを作る変換器の例。核の外・使い捨て想定。

眠気(KSS)や行動クラスの例と違い、ストレスは映像では付けにくいので生体信号から写す。
流れは「波形 → 拍検出 → RMSSD → 段階」。実データ(UBFC-PHYS 等)では「PPG波形と
サンプリング周波数を読む部分」を配布形式に合わせて書き換える。ここは実データが無いので
ダミーのPPGを合成して流れを示す。段階のしきい値(RMSSDの降順境界)＝アノテ規約はここに明示。
"""

import numpy as np

from alertness.bio import detect_peaks, rmssd, rr_intervals_ms, stage_from_rmssd
from alertness.ingest.mapping import segment, write_manifest

FS = 64.0  # PPG のサンプリング周波数[Hz]（データセット固有。ここではダミー）
WINDOW_S = 30.0  # ラベルを付ける時間窓[秒]
# アノテ規約: RMSSD[ms] の降順しきい値。高い＝落ち着き(none)、低い＝ストレス(high)。
RMSSD_THRESHOLDS = (50.0, 35.0, 20.0)


def _render_ppg(beat_times_s: np.ndarray, seconds: float, fs: float) -> np.ndarray:
    # 各拍の時刻にガウス状の脈波を置いた擬似 PPG を作る。
    t = np.arange(0, seconds, 1.0 / fs)
    signal = np.zeros_like(t)
    width = 0.05  # 脈波の幅[秒]
    for bt in beat_times_s:
        signal += np.exp(-0.5 * ((t - bt) / width) ** 2)
    return signal


def _dummy_beats(hr_bpm: float, jitter_s: float, seconds: float, seed: int) -> np.ndarray:
    # 平均 RR に jitter を足した拍列。jitter が小さいほど RMSSD が下がる（ストレス側）。
    rng = np.random.default_rng(seed)
    mean_rr = 60.0 / hr_bpm
    times, t = [], mean_rr
    while t < seconds:
        times.append(t)
        t += mean_rr + rng.normal(0.0, jitter_s)
    return np.array(times)


# データセット固有: 生波形の読み取り（ここではダミー）。窓ごとに jitter を変えて
# 「落ち着き→ストレス」を作る。実データではこの3行を PPG 読み込みに置き換える。
windows = [
    _render_ppg(_dummy_beats(72, jitter, WINDOW_S, seed), WINDOW_S, FS)
    for seed, jitter in enumerate((0.06, 0.03, 0.012))
]

segments = []
for i, ppg in enumerate(windows):
    peaks = detect_peaks(ppg, FS)
    stage = stage_from_rmssd(rmssd(rr_intervals_ms(peaks / FS)), RMSSD_THRESHOLDS)
    start = i * WINDOW_S
    segments.append(segment(start, start + WINDOW_S, stress=stage))

path = write_manifest(
    "data/manifests/example_stress_s01.json",
    video="s01.mp4",
    subject="s01",
    context="driving",
    segments=segments,
)
print(f"manifest を書き出しました: {path}")
print("各窓のストレス段階:", [s.get("stress") for s in segments])
