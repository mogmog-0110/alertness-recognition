"""閉眼の手がかり（PERCLOS）。眠気の主要シグナル。"""

from __future__ import annotations

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._episodes import closure_episodes
from ._eye_health import eye_signal_usable
from ._support import (
    eye_key,
    recency_weight,
    time_fraction,
    trailing_true_seconds,
    window_coverage,
    window_values,
)


class EyeClosureCue:
    name = "eye_closure"
    dimension = "drowsiness"

    def __init__(
        self,
        window_seconds: float = 30.0,
        perclos_drowsy: float = 0.4,
        closed_ratio: float = 0.6,
        max_yaw: float = 25.0,
        min_coverage: float = 0.5,
        health_window: float = 60.0,
        min_window: float = 10.0,
        release_half_life: float = 2.0,
        long_closure_seconds: float = 0.5,
    ) -> None:
        self.window_seconds = window_seconds
        self.perclos_drowsy = perclos_drowsy  # この割合以上閉じていたら眠気とみなす
        self.closed_ratio = closed_ratio  # 開眼基準の何割未満で閉眼とするか
        self.max_yaw = max_yaw  # これ以上横を向くとEARが信用できないので判定しない
        self.min_coverage = min_coverage  # 窓のうち顔が見えていた時間がこれ未満なら判定しない
        self.health_window = health_window  # この長さに瞬きが1回も無ければ目の信号を信じない
        # 履歴がこれだけ貯まるまでは判定しない。測り直しで履歴を捨てた直後は数百 ms しか無く、
        # 瞬き 1 回で閉眼の割合が 3 割を超えて眠気の警告になる。
        self.min_window = min_window
        # 長い閉眼が終わって目を開けていると、この半減期で点を下げる。PERCLOS は 30 秒の平均
        # なので、そのままでは目を大きく開けても閉眼が窓から抜けるまで警告が残る。直したのに
        # 鳴り続ける警告は、警告として働かない。普通の瞬き（long_closure_seconds 未満）は
        # 「閉じた」に数えないので、瞬きのたびに点が戻ることはない。
        self.release_half_life = release_half_life
        self.long_closure_seconds = long_closure_seconds

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし", None, False)
        if abs(obs.features.get("yaw_rel", 0.0)) > self.max_yaw:
            # 横顔ではEARが壊れて誤検出するので、眠気判定から除外する。
            return CueResult(self.name, self.dimension, 0.0, False, "横向き", None, False)

        coverage = window_coverage(obs, self.window_seconds)
        if coverage < self.min_coverage:
            # 窓の大半で顔を見失っている。残った少数のフレームで出した PERCLOS は、
            # 窓全体の閉眼割合を表していない。
            detail = f"計測不足 {coverage:.0%}"
            return CueResult(self.name, self.dimension, 0.0, False, detail, None, False)

        usable, reason = eye_signal_usable(obs, self.health_window, closed_ratio=self.closed_ratio)
        if not usable:
            # サングラス・暗所では EAR が低いまま張り付き、閉じていないのに PERCLOS が
            # 上がる。眠気を誤って警告する向きの誤りなので、黙って頭部の cue に譲る。
            return CueResult(self.name, self.dimension, 0.0, False, reason, None, False)

        times, ears = window_values(obs, eye_key(obs), self.window_seconds, 1.0)
        if not times or times[-1] - times[0] < self.min_window:
            return CueResult(self.name, self.dimension, 0.0, False, "閉眼を観察中", None, False)
        flags = [e < self.closed_ratio for e in ears]
        perclos = time_fraction(times, flags)
        freshness = recency_weight(self._open_for(times, ears), self.release_half_life)
        score = clamp(perclos / self.perclos_drowsy) if self.perclos_drowsy > 0 else 0.0
        active = perclos >= self.perclos_drowsy and freshness >= 0.5
        return CueResult(
            self.name, self.dimension, score * freshness, active, f"PERCLOS {perclos:.2f}"
        )

    def _open_for(self, times: list[float], ears: list[float]) -> float:
        """最後の長い閉眼が終わってから目を開けている秒数。長く閉じている最中は 0。

        いま閉じていても long_closure_seconds に満たなければ普通の瞬きとして開いている側に
        数える。瞬きのたびに 0 へ戻すと、下げた点が瞬きの間だけ満点に戻って警告が立つ。
        窓に長い閉眼が無いのに PERCLOS が高いとき（半目が続く・細かく閉じ続ける）は、
        目を開けたとは言えないので 0 を返し、点を下げない。
        """
        opened = self.closed_ratio + 0.1
        closing = trailing_true_seconds(times, [e < opened for e in ears])
        if not ears or closing >= self.long_closure_seconds:
            return 0.0
        closures = closure_episodes(times, ears, self.closed_ratio, opened)
        long = [c for c in closures if c.duration >= self.long_closure_seconds]
        return times[-1] - long[-1].end if long else 0.0
