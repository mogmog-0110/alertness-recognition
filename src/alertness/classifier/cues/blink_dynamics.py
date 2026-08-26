"""瞬きの遅さの手がかり。眠気の「早期」シグナル。

既存の2本は眠気の両端しか見ていない。PERCLOS は 30 秒窓で平均するので反応が遅く、
1 秒の閉眼（blink cue）が出たときはもうマイクロスリープに入っている。その間を埋めるのが
瞬きの動き方で、眠気が進むと次の順で変わる:
- 1回の閉眼が長くなる（覚醒時 0.1〜0.15 秒 → 眠気で 0.3 秒以上）
- まぶたを持ち上げる動きが遅くなる（閉じきってから開くまでが伸びる）

長さと戻りの遅さは同じ生理（まぶたを持ち上げる筋の弛緩）から来るので、足し合わせずに
大きい方を採る。足すと同じ現象を二重に数えて、片方だけでも警告に届いてしまう。

しきい値は文献の代表値を初期値に置いてある。個人差があるので、収録データで詰めること。
"""

from __future__ import annotations

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._episodes import closure_episodes
from ._support import window_coverage, window_values


class BlinkDynamicsCue:
    name = "blink_dynamics"
    dimension = "drowsiness"

    def __init__(
        self,
        window_seconds: float = 60.0,
        closed_ratio: float = 0.6,
        open_ratio: float = 0.7,
        normal_seconds: float = 0.15,
        drowsy_seconds: float = 0.35,
        normal_reopen: float = 0.12,
        drowsy_reopen: float = 0.30,
        min_blinks: int = 3,
        max_yaw: float = 25.0,
        min_coverage: float = 0.5,
    ) -> None:
        self.window_seconds = window_seconds  # 瞬きを集める窓
        self.closed_ratio = closed_ratio  # 開眼基準の何割を下回ったら閉じ始めか
        self.open_ratio = open_ratio  # 何割を上回ったら開き始めか（入口より高くする）
        self.normal_seconds = normal_seconds  # 覚醒時の閉眼時間。ここまでは 0 点
        self.drowsy_seconds = drowsy_seconds  # ここまで伸びたら満点
        self.normal_reopen = normal_reopen  # 覚醒時の戻り時間
        self.drowsy_reopen = drowsy_reopen  # ここまで遅れたら満点
        self.min_blinks = min_blinks  # これだけ瞬きを見ないと平均を出さない
        self.max_yaw = max_yaw  # 横顔では EAR が壊れるので判定しない
        self.min_coverage = min_coverage  # 窓のうち顔が見えていた時間の下限

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし", None, False)
        if abs(obs.features.get("yaw_rel", 0.0)) > self.max_yaw:
            return CueResult(self.name, self.dimension, 0.0, False, "横向き", None, False)

        coverage = window_coverage(obs, self.window_seconds)
        if coverage < self.min_coverage:
            detail = f"計測不足 {coverage:.0%}"
            return CueResult(self.name, self.dimension, 0.0, False, detail, None, False)

        times, ears = window_values(obs, "ear_norm", self.window_seconds, 1.0)
        episodes = closure_episodes(times, ears, self.closed_ratio, self.open_ratio)
        if len(episodes) < self.min_blinks:
            # 瞬きが少ないうちに平均を出すと、1回の外れがそのまま判定になる。
            detail = f"瞬き {len(episodes)}回（観察中）"
            return CueResult(self.name, self.dimension, 0.0, False, detail, None, False)

        duration = _mean(e.duration for e in episodes)
        reopens = [e.reopen_seconds for e in episodes if e.reopen_seconds is not None]
        by_duration = _ramp(duration, self.normal_seconds, self.drowsy_seconds)
        by_reopen = 0.0
        if reopens:
            by_reopen = _ramp(_mean(reopens), self.normal_reopen, self.drowsy_reopen)

        score = max(by_duration, by_reopen)
        detail = f"閉眼 {duration * 1000:.0f}ms 戻り {_ms(reopens)}"
        return CueResult(self.name, self.dimension, score, score >= 1.0, detail)


def _ramp(value: float, low: float, high: float) -> float:
    """low で 0 点、high で満点になる直線。幅が無ければ 0。"""
    width = high - low
    return clamp((value - low) / width) if width > 0 else 0.0


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _ms(values: list[float]) -> str:
    return f"{_mean(values) * 1000:.0f}ms" if values else "—"
