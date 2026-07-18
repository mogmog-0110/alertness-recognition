"""HRV 指標をストレスの段階ラベル(none/low/medium/high)へ写す。

前提: ストレスが上がると副交感神経の活動が下がり、RMSSD などの短期変動が小さくなる。
つまり RMSSD が高い＝落ち着き(none)、低い＝ストレス(high)という向き。ordinal_bin が
「値が高いほど段階が上」なのに対し、ここは逆向き（値が低いほど段階が上）になる。

しきい値は被験者・計測条件に強く依存する（安静時の RMSSD 自体が個人差大）。そのため
絶対値でなく、各データセット/被験者の基準に合わせて呼び出し側で決める（ここでは写像だけ）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..ingest.manifest import LEVELS


def stage_from_rmssd(
    rmssd_ms: float,
    thresholds: Sequence[float],
    levels: Sequence[str] = LEVELS,
) -> str:
    """RMSSD[ミリ秒] → 段階。thresholds は降順（例: [50, 35, 20]）。

    値が高い（変動大＝落ち着き）ほど軽い段階へ。thresholds=[t1,t2,t3] のとき:
      rmssd >= t1 → none, t2..t1 → low, t3..t2 → medium, t3 未満 → high。
    値が nan（拍が足りず計算不能）のときは、段階を断定せず空文字（未アノテ）を返す。
    """
    if len(thresholds) != len(levels) - 1:
        raise ValueError("thresholds は levels より1つ少ない必要があります。")
    if any(a < b for a, b in zip(thresholds, thresholds[1:], strict=False)):
        raise ValueError("thresholds は降順である必要があります（高い RMSSD ほど落ち着き）。")
    if rmssd_ms is None or math.isnan(rmssd_ms):
        return ""
    for level, t in zip(levels, thresholds, strict=False):
        if rmssd_ms >= t:
            return level
    return levels[-1]
