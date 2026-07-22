"""集中の手がかり（走査）。視線が動いているか＝周囲を見ているかを見る。

視線が1点に留まっていることは集中ではない。運転では危険予測のために複数方向を走査する
必要があり、実証的にも「熟練者ほど水平方向の走査が広く、初心者ほど狭い」ことが繰り返し
確認されている。訓練で走査の広がりが増えると危険検出も改善する。逆に認知負荷がかかると
視線が前方中心に集まる（視野狭窄）ので、走査の少なさは認知的な注意逸脱の兆候になる。

指標は PRC（Percent Road Center）＝窓の中で視線が中心域に留まっていた割合。視覚的・
認知的負荷に対して、注視やサッケードに基づく指標より感度が高く頑健とされる。補助として
水平視線のばらつきも見る（認知負荷に対して最も感度が高く、かつ計算が簡単な指標）。

対になる attention_buffer が「目が対象から離れすぎていないか」を見るのに対し、こちらは
「目が対象に貼りついたまま動いていないか」を見る。逆向きの失敗を1本ずつ担当する。
"""

from __future__ import annotations

import math

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._support import window_values


class GazeScanningCue:
    name = "gaze_scanning"
    dimension = "concentration"

    def __init__(
        self,
        window_seconds: float = 30.0,
        center_radius: float = 0.035,
        prc_healthy: float = 0.85,
        prc_frozen: float = 1.0,
        min_spread: float = 0.004,
        min_window: float = 10.0,
    ) -> None:
        self.window_seconds = window_seconds  # 走査を見る窓
        self.center_radius = center_radius  # 中心域とみなす視線ズレの半径
        # PRC がこれを超えると走査不足として点が下がり始める。文献の 92% は専用の
        # アイトラッカーと車内での道路中心域の定義に基づく値で、カメラ位置も尺度も違う
        # この実装にそのままは移せない。自前のデータで較正し直すこと。
        self.prc_healthy = prc_healthy
        self.prc_frozen = prc_frozen  # ここまで来ると 0 点（完全に貼りついている）
        self.min_spread = min_spread  # 水平視線のばらつきがこれ未満なら走査していない
        self.min_window = min_window  # これだけ履歴が無いと判定しない

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 1.0, False, "顔なし", None, False)

        times, raw = window_values(obs, "gaze_off", self.window_seconds, float("nan"))
        pairs = [(t, v) for t, v in zip(times, raw, strict=False) if not math.isnan(v)]
        span = pairs[-1][0] - pairs[0][0] if len(pairs) > 1 else 0.0
        if len(pairs) < 5 or span < self.min_window:
            # 履歴が浅いうちは判定しない。走査は数十秒の窓で見る性質の指標なので、
            # 起動直後を「走査していない」と読むと必ず誤警告になる。
            return CueResult(self.name, self.dimension, 1.0, False, "走査を観察中", None, False)

        values = [v for _, v in pairs]
        prc = sum(1 for v in values if v <= self.center_radius) / len(values)
        spread = _stdev(values)
        score = min(self._prc_score(prc), self._spread_score(spread))
        detail = f"PRC {prc:.0%} 視線ばらつき {spread:.4f}"
        return CueResult(self.name, self.dimension, score, score < 0.5, detail, None, True)

    def _prc_score(self, prc: float) -> float:
        # 中心域に留まりすぎているほど下げる。healthy 以下なら満点。
        width = self.prc_frozen - self.prc_healthy
        if width <= 0:
            return 1.0
        return 1.0 - clamp((prc - self.prc_healthy) / width)

    def _spread_score(self, spread: float) -> float:
        # ばらつきが min_spread に届かないほど下げる。PRC と食い違うときは低い方を採る。
        return clamp(spread / self.min_spread) if self.min_spread > 0 else 1.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
