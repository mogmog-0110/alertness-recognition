"""顔を見失う直前に、どちらを向いていたかを見る共通部品。

横を大きく向いたり手元を深く見たりすると、顔の検出そのものが外れて向きが測れなくなる。
そのまま「見失った」と扱うと、脇見なのに眠気（face_absent）として鳴る。見失う直前に目を
開けたまま大きく向きを変えていたなら、見失いはその動きの続きとして読む。

目が閉じていたら読み替えない。目を閉じたまま前や横に崩れて顔が外れるのは、居眠りで
最も危ない形で、face_absent が受け持つべきものだから。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import median

from ...contracts import Features, Observation

TURN = "turn"
DOWN = "down"


@dataclass(frozen=True)
class Departure:
    kind: str  # TURN（横を向いて外れた）か DOWN（下を向いて外れた）
    absent_seconds: float  # 最後に顔が見えてからの秒数
    moved_seconds: float  # 見失う前に向きを変えていた秒数


def departure(
    obs: Observation,
    yaw_deg: float,
    pitch_deg: float,
    window_seconds: float = 15.0,
    lookback_seconds: float = 1.0,
    open_ratio: float = 0.6,
    eyes_seconds: float = 0.3,
) -> Departure | None:
    """いま顔を見失っていて、その直前に大きく向きを変えていたなら Departure。

    yaw_deg / pitch_deg が 0 以下の向きは見ない。lookback_seconds は見失う直前のどこまでを
    「直前の向き」とみなすか。速く向きを変えると、最後の 1 フレームはまだ途中の角度になる。
    目の状態は、向きを変え始める直前 eyes_seconds で見る。下を向くとまぶたも一緒に下がる
    ので、見失う直前の目で見ると手元を見ただけでも「目を閉じて崩れた」と読んでしまう。
    居眠りで崩れるときは、頭が落ちる前に目が閉じている。
    """
    frames = list(obs.history.recent(window_seconds))
    if not frames or frames[-1].face_present:
        return None
    last = _last_present(frames)
    if last is None:
        return None
    seen = [f for f in frames[: last + 1] if f.face_present]
    seen_at = seen[-1].timestamp
    absent = frames[-1].timestamp - seen_at
    def turned(f: Features) -> bool:
        return abs(f.get("yaw_rel", 0.0)) >= yaw_deg

    def lowered(f: Features) -> bool:
        return f.get("pitch_rel", 0.0) >= pitch_deg

    for kind, degrees, moved in ((TURN, yaw_deg, turned), (DOWN, pitch_deg, lowered)):
        if degrees <= 0:
            continue
        began = _move_began(seen, moved, seen_at - lookback_seconds)
        if began is None:
            continue
        eyes = _eyes([f for f in seen if began - eyes_seconds <= f.timestamp < began])
        if eyes < open_ratio:
            return None
        return Departure(kind, absent, seen_at - began)
    return None


_Moved = Callable[[Features], bool]


def _last_present(frames: list[Features]) -> int | None:
    for i in range(len(frames) - 1, -1, -1):
        if frames[i].face_present:
            return i
    return None


def _move_began(seen: list[Features], moved: _Moved, earliest: float) -> float | None:
    """earliest 以降に向きを変えていたフレームがあれば、その動きが始まった時刻。"""
    recent = [i for i in range(len(seen)) if seen[i].timestamp >= earliest and moved(seen[i])]
    if not recent:
        return None
    index = recent[-1]
    while index > 0 and moved(seen[index - 1]):
        index -= 1
    return seen[index].timestamp


def _eyes(frames: list[Features]) -> float:
    key = "eye_open" if any("eye_open" in f.values for f in frames) else "ear_norm"
    values = [f.get(key) for f in frames if f.get(key) == f.get(key)]
    return median(values) if values else 1.0
