"""運転者を見失っている手がかり。

他の cue はすべて「顔なし」で黙る（score 0）。そのため顔が取れない間、判定はどの軸も
出ず、画面上は「異常なし」と区別がつかない。停まって突っ伏した、体が大きく崩れた、
カメラが塞がれた——どれも危険側の出来事なのに、無言になるのが今までの振る舞いだった。

この cue だけは顔が無いときに働く。眠気の軸に置いてあるのは、運転席で顔が数秒消える
物理的な原因（前に倒れる・横に崩れる）が眠気側に寄っているため。カメラの遮蔽や
取り付けのずれでも立つので、detail に理由を書いて切り分けられるようにする。
"""

from __future__ import annotations

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._departure import TURN, departure
from ._support import trailing_true_seconds


class FaceAbsentCue:
    name = "face_absent"
    dimension = "drowsiness"

    def __init__(
        self,
        absent_seconds: float = 3.0,
        grace_seconds: float = 0.5,
        explain_yaw_deg: float = 0.0,
        explain_pitch_deg: float = 0.0,
        explained_hold_seconds: float = 10.0,
    ) -> None:
        self.absent_seconds = absent_seconds  # 連続でこれだけ見失ったら満点
        self.grace_seconds = grace_seconds  # 一瞬の検出漏れは数えない猶予
        # 目を開けたままこの角度以上横／下を向いた直後に見失ったのは、脇見・手元の注視の
        # 続き。head_turn と attention_buffer に任せ、explained_hold_seconds までは黙る。
        # 0 ならその向きは見ない。
        self.explain_yaw_deg = explain_yaw_deg
        self.explain_pitch_deg = explain_pitch_deg
        self.explained_hold_seconds = explained_hold_seconds

    def evaluate(self, obs: Observation) -> CueResult:
        if obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "")
        gone = departure(obs, self.explain_yaw_deg, self.explain_pitch_deg)
        if gone is not None and gone.absent_seconds <= self.explained_hold_seconds:
            detail = "横を向いて顔が外れた" if gone.kind == TURN else "下を向いて顔が外れた"
            return CueResult(self.name, self.dimension, 0.0, False, detail)

        window = max(2.0, self.absent_seconds * 2)
        frames = obs.history.recent(window)
        times = [f.timestamp for f in frames]
        absent = trailing_true_seconds(times, [not f.face_present for f in frames])
        counted = max(0.0, absent - self.grace_seconds)
        span = max(1e-6, self.absent_seconds - self.grace_seconds)
        score = clamp(counted / span)
        active = absent >= self.absent_seconds
        detail = f"運転者を検出できず {absent:.1f}s"
        return CueResult(self.name, self.dimension, score, active, detail)
