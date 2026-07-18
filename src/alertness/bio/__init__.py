"""生体信号から教師ラベルを作るための汎用部品（numpy のみ、重い依存なし）。

役割は「波形→拍→HRV指標→段階ラベル」の各段を、データセット非依存の純関数で提供すること。
配布フォーマットの読み取り（どの列がPPGか、サンプリング周波数はいくつか 等）は核の外
（examples/convert_*.py）に置く。ここはその読み取り結果を受け取って段階ラベルへ写すだけ。
"""

from __future__ import annotations

from .hrv import mean_hr, pnn50, rmssd, rr_intervals_ms, sdnn
from .peaks import detect_peaks
from .stress import stage_from_rmssd

__all__ = [
    "detect_peaks",
    "rr_intervals_ms",
    "mean_hr",
    "sdnn",
    "rmssd",
    "pnn50",
    "stage_from_rmssd",
]
