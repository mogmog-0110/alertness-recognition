"""cue と policy を合成する判定器。Classifier ポートの実装。

ここはあくまで合成だけ。将来 ML/DL を入れるときは、この CueClassifier の隣に
別の Classifier 実装（features や history を直接読むもの）を置けばよく、
本体（パイプライン）は無修正で差し替えられる。
"""

from __future__ import annotations

from collections.abc import Sequence

from ..contracts import Assessment, Observation
from ..ports import Cue, DecisionPolicy, Resettable


class CueClassifier:
    def __init__(self, cues: Sequence[Cue], policy: DecisionPolicy) -> None:
        self._cues = tuple(cues)
        self._policy = policy

    def assess(self, obs: Observation) -> Assessment:
        results = tuple(c.evaluate(obs) for c in self._cues)
        return self._policy.decide(obs, results)

    def reset(self) -> None:
        """状態を持つ cue と policy をまとめて初期化する。

        再キャリブレーションは「別人になったかもしれない」の合図なので、本人の安静を
        前提に育てた基準（心拍・表情）と、積み上げた注意残高・平滑値をここで捨てる。
        """
        for part in (*self._cues, self._policy):
            if isinstance(part, Resettable):
                part.reset()
