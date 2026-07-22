"""ストレスの手がかり（表情）。眉を寄せ、まぶたを緊張させる動きの強さを見る。

皺眉筋（眉を下げる筋、AU4）の活動は、心理生理学で最も長く使われてきた指標の一つで、
負の感情価と認知的な努力の両方を反映する。前部帯状回の制御下にあり、葛藤・痛み・
認知制御に応じて活動が上がることが繰り返し確認されている。まぶたの緊張（AU7）も、
心理社会的ストレスの複数の指標と関連することが報告されていて、多指標での判定に使える。

MediaPipe の blendshape は FACS の AU に対応づけられている（臨床心理士10名による検証で
88%が全員一致）。ここで使うのは browDown→AU4、eyeSquint→AU7、mouthPress→AU24。

注意すべき性質が2つある。どちらも「本人の安静との差で見る」ことで扱う:
- 個人差が大きい。表情筋の指標は絶対値では比較できず、被験者ごとの正規化が標準の作法。
- 向きが一様でない。ストレス下で皺眉筋が上がる人と下がる人がいるという報告がある。
  よって、この cue 単独ではストレスを断定しない（active を立てない）。心拍と揃って
  初めて効くよう、軸側の統合を weighted にして使うことを想定している。
"""

from __future__ import annotations

import math

from ...contracts import CueResult, Observation
from ._baseline import AdaptiveBaseline
from ._support import window_values

# 使う blendshape と、対応する AU。左右は平均して1つの値にする。
_TENSION_PAIRS = (
    ("browDownLeft", "browDownRight"),  # AU4 眉を下げる（皺眉筋）
    ("eyeSquintLeft", "eyeSquintRight"),  # AU7 まぶたを緊張させる
    ("mouthPressLeft", "mouthPressRight"),  # AU24 唇を押しつける
)


class FacialTensionCue:
    name = "facial_tension"
    dimension = "stress"

    def __init__(
        self,
        span: float = 0.15,
        sustained_seconds: float = 5.0,
        baseline_seconds: float = 120.0,
        min_rest_samples: int = 30,
        rest_interval: float = 1.0,
        freeze_at: float = 0.3,
        noise_k: float = 6.0,
    ) -> None:
        self.span = span  # 基準からこれだけ上がると満点（blendshape は 0..1）
        self.sustained_seconds = sustained_seconds  # 現在値を取る窓
        self._baseline = AdaptiveBaseline(
            seconds=baseline_seconds,
            min_samples=min_rest_samples,
            interval=rest_interval,
            freeze_at=freeze_at,
            noise_k=noise_k,
        )

    def evaluate(self, obs: Observation) -> CueResult:
        progress = self._baseline.progress()
        if not obs.features.face_present:
            self._baseline.reset()
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし", progress, False)

        current = self._tension(obs)
        if current is None:
            detail = "表情の値なし"  # blendshape 無効時。ストレスを none と断定しない。
            return CueResult(self.name, self.dimension, 0.0, False, detail, progress, False)

        base, spread, ready = self._baseline.read()
        score = self._baseline.score(current - base, self.span, spread) if ready else 0.0
        self._baseline.update(obs.features.timestamp, current, ready, score)
        if not ready:
            detail = "表情の基準を測定中"
            return CueResult(self.name, self.dimension, 0.0, False, detail, progress, False)

        detail = f"表情緊張 {current:.2f} base {base:.2f}"
        # active は立てない。単独でストレスを断定できる指標ではなく、心拍を補強する役。
        return CueResult(self.name, self.dimension, score, False, detail, progress, True)

    def _tension(self, obs: Observation) -> float | None:
        """直近の窓での緊張の強さ。左右を平均し、AU をまたいで平均する。"""
        values = []
        for left, right in _TENSION_PAIRS:
            pair = [self._mean(obs, key) for key in (left, right)]
            usable = [v for v in pair if v is not None]
            if usable:
                values.append(sum(usable) / len(usable))
        return sum(values) / len(values) if values else None

    def _mean(self, obs: Observation, key: str) -> float | None:
        window = max(2.0, self.sustained_seconds)
        _, raw = window_values(obs, key, window, float("nan"))
        clean = [v for v in raw if not math.isnan(v)]
        return sum(clean) / len(clean) if clean else None
