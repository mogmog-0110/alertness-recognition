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
from ._support import eye_key, recency_weight, window_coverage, window_values
from ._support import weighted_median as _weighted_median


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
        half_life_seconds: float = 20.0,
        max_blink_seconds: float = 2.0,
        release_half_life: float = 3.0,
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
        # 古い瞬きの重みが半分になるまでの時間。素の平均だと 1 回の長い瞬きが
        # 窓の長さぶん効き続け、直後に普通の瞬きを重ねても下がりきらない。
        self.half_life_seconds = half_life_seconds
        self.min_coverage = min_coverage  # 窓のうち顔が見えていた時間の下限
        # これより長い閉眼は瞬きではなく、意図して閉じたかマイクロスリープ。blink cue が
        # その場で受け持つ。瞬きに混ぜると、目を開けた後も窓に残る数回の中央値を押し上げ、
        # 開けてから数秒「まばたきが遅い」が立ち続ける。
        self.max_blink_seconds = max_blink_seconds
        # 最後の遅い瞬きからの経過でこの半減期で点を下げる。窓の中央値は、遅い瞬きを続けた
        # 後だと普通の瞬きが同じ数だけ貯まるまで動かず、目を開けて普通に瞬きしていても
        # 数十秒「まばたきが遅い」が立ち続ける。眠い間は遅い瞬きが数秒おきに続くので下がらない。
        self.release_half_life = release_half_life

    def evaluate(self, obs: Observation) -> CueResult:
        if not obs.features.face_present:
            return CueResult(self.name, self.dimension, 0.0, False, "顔なし", None, False)
        if abs(obs.features.get("yaw_rel", 0.0)) > self.max_yaw:
            return CueResult(self.name, self.dimension, 0.0, False, "横向き", None, False)

        coverage = window_coverage(obs, self.window_seconds)
        if coverage < self.min_coverage:
            detail = f"計測不足 {coverage:.0%}"
            return CueResult(self.name, self.dimension, 0.0, False, detail, None, False)

        times, ears = window_values(obs, eye_key(obs), self.window_seconds, 1.0)
        episodes = [
            e
            for e in closure_episodes(times, ears, self.closed_ratio, self.open_ratio)
            if e.duration <= self.max_blink_seconds
        ]
        if len(episodes) < self.min_blinks:
            # 瞬きが少ないうちに平均を出すと、1回の外れがそのまま判定になる。
            detail = f"瞬き {len(episodes)}回（観察中）"
            return CueResult(self.name, self.dimension, 0.0, False, detail, None, False)

        # 中央値を使う。平均だと外れ値 1 個で跳ねる (実測: 中央値 100ms のところ
        # 平均が 422ms まで上がり、眠気の警告が立ち続けた)。窓に入る瞬きは数回
        # しかないので min_blinks では守れない。あわせて古い瞬きほど軽く扱う。
        now = times[-1] if times else 0.0
        weights = [recency_weight(now - e.end, self.half_life_seconds) for e in episodes]
        duration = _weighted_median([e.duration for e in episodes], weights)
        reopen_pairs = [
            (e.reopen_seconds, w)
            for e, w in zip(episodes, weights, strict=True)
            if e.reopen_seconds is not None
        ]
        by_duration = _ramp(duration, self.normal_seconds, self.drowsy_seconds)
        by_reopen = 0.0
        if reopen_pairs:
            by_reopen = _ramp(
                _weighted_median([v for v, _ in reopen_pairs], [w for _, w in reopen_pairs]),
                self.normal_reopen,
                self.drowsy_reopen,
            )

        raw = max(by_duration, by_reopen)
        freshness = recency_weight(now - self._last_slow(episodes, now), self.release_half_life)
        detail = f"閉眼 {duration * 1000:.0f}ms 戻り {_ms([v for v, _ in reopen_pairs])}"
        active = raw >= 1.0 and freshness >= 0.5
        return CueResult(self.name, self.dimension, raw * freshness, active, detail)

    def _last_slow(self, episodes: list, now: float) -> float:
        """普通より長いか戻りの遅い瞬きのうち、最後のものが終わった時刻。無ければ now。"""
        slow = [
            e.end
            for e in episodes
            if e.duration > self.normal_seconds
            or (e.reopen_seconds is not None and e.reopen_seconds > self.normal_reopen)
        ]
        return slow[-1] if slow else now


def _ramp(value: float, low: float, high: float) -> float:
    """low で 0 点、high で満点になる直線。幅が無ければ 0。"""
    width = high - low
    return clamp((value - low) / width) if width > 0 else 0.0


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _ms(values: list[float]) -> str:
    return f"{_mean(values) * 1000:.0f}ms" if values else "—"
