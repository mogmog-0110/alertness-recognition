"""連続眠気スコア（CDS）を4段階の Level of Drowsiness（LoD）へ対応付ける。

``calibrate_thresholds`` は、PVT1に対する反応時間・lapse率の悪化量を使い、眠気が悪化した
セッションほど段階境界を設定範囲内で下げる。``classify_lod`` は3つの昇順境界によりCDSを
none/low/medium/highへ離散化する。CDS自体の算出やPVT集計はこのモジュールの責務ではない。

DROZY manifest経路では校正済み境界を時間平滑化処理へ渡し、短い遷移を除いてから区間化する。
既定境界は単体利用向けであり、データ変換時は ``config/default.yaml`` の ``drozy.lod`` を使う。
"""

from __future__ import annotations

from collections.abc import Sequence


def calibrate_thresholds(
    thresholds: Sequence[float],
    impairment: float,
    *,
    gain: float = 5.0,
    max_shift: float = 10.0,
) -> tuple[float, float, float]:
    """PVT1から悪化したセッションではLoD境界を設定範囲内で下げる。"""
    if len(thresholds) != 3:
        raise ValueError("thresholds は3値である必要があります")
    shift = max(-max_shift, min(max_shift, float(impairment) * gain))
    return tuple(float(value) - shift for value in thresholds)  # type: ignore[return-value]


def classify_lod(
    scores: Sequence[float], *, thresholds: Sequence[float] = (20.0, 50.0, 75.0)
) -> list[str]:
    """CDS を None / Low / Medium / High に変換する。"""
    if len(thresholds) != 3 or list(thresholds) != sorted(thresholds):
        raise ValueError("thresholds は昇順の3値である必要があります")
    low, medium, high = (float(value) for value in thresholds)
    levels: list[str] = []
    for score in scores:
        if score < low:
            levels.append("none")
        elif score < medium:
            levels.append("low")
        elif score < high:
            levels.append("medium")
        else:
            levels.append("high")
    return levels
