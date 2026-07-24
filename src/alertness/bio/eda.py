"""EDA(皮膚電気活動)から覚醒度を出し、ストレスの段階ラベルへ写す。

前提: 交感神経が高ぶると発汗が増え、皮膚コンダクタンス(tonic SCL)が上がる。心拍変動と違い
体動に強く（発汗はゆっくり変わる）、UBFC-Phys の実測では 18人中16人が安静(T1)より
ストレスタスク(T2/T3)で SCL が上がった（心拍は 5/18 しか一致しなかった）。

SCL の絶対値は個人差が桁で違う（安静SCLが 0.05 の人と 1.3 の人がいる）ので、被験者ごとに
安静を基準にし、その人の EDA の振れ幅で正規化した「安静からどれだけ上がったか(0..1)」に
そろえる。段階のしきい値はこの相対値に対して置く。窓ごとに写すので時間変化を持つ。

しきい値は呼び出し側で決める（ここは写像だけ）。段階を細かく分けるより、落ち着き/上昇の
2値に束ねた方が実データで安定する（EDAの4段階は境界が恣意的で精度が出ない）。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..ingest.manifest import LEVELS


def tonic_windows(
    signal: Sequence[float], fs: float, window_seconds: float
) -> list[tuple[float, float]]:
    """波形を窓に区切り、(窓中央の時刻[秒], 窓内の中央値SCL) の並びを返す。

    中央値を使うのは、発汗の急峻な応答(SCR)や外れ値に引かれず tonic な水準を見るため。
    端数のフレームは切り捨てる。
    """
    values = np.asarray(signal, dtype=float).ravel()
    width = int(round(window_seconds * fs))
    if width < 1 or fs <= 0:
        return []
    out = []
    for start in range(0, values.size - width + 1, width):
        center = (start + width / 2) / fs
        out.append((center, float(np.median(values[start : start + width]))))
    return out


def subject_scale(
    rest_windows: Sequence[tuple[float, float]],
    all_windows: Sequence[tuple[float, float]],
    min_spread: float = 1e-3,
) -> tuple[float, float] | None:
    """安静基準SCLと、その人のEDAの振れ幅(spread)を出す。窓が無ければ None。

    基準は安静(T1)窓の中央値。spread は全タスクの窓中央値の P10-P90（その人がどれだけ
    EDA を動かすか）。反応が乏しい人で spread が 0 に潰れないよう下限を掛ける。
    """
    rest = [scl for _, scl in rest_windows]
    every = [scl for _, scl in all_windows]
    if not rest or not every:
        return None
    baseline = float(np.median(rest))
    spread = float(np.percentile(every, 90) - np.percentile(every, 10))
    return baseline, max(spread, min_spread)


def relative_arousal(scl: float, baseline: float, spread: float) -> float:
    """安静からの上昇を、その人の振れ幅で割った覚醒度。0=安静水準、1=よく上がった状態。"""
    return (scl - baseline) / spread if spread > 0 else 0.0


def stage_from_arousal(
    arousal: float, thresholds: Sequence[float], levels: Sequence[str] = LEVELS
) -> str:
    """覚醒度(0..1)を段階へ。thresholds は昇順（例: [0.15, 0.4, 0.7]）。

    arousal < t1 → none, t1..t2 → low, t2..t3 → medium, t3 以上 → high。
    ordinal_bin と同じ向き（値が高いほど段階が上）。
    """
    if len(thresholds) != len(levels) - 1:
        raise ValueError("thresholds は levels より1つ少ない必要があります。")
    if any(a > b for a, b in zip(thresholds, thresholds[1:], strict=False)):
        raise ValueError("thresholds は昇順である必要があります。")
    for level, t in zip(levels, thresholds, strict=False):
        if arousal < t:
            return level
    return levels[-1]
