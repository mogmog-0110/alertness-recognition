"""cue の結果を評価軸ごとに統合するルールベースの方針。

軸ごとに「重み付き平均」と「最も強い単独シグナル」の大きい方を取る。
弱い手がかりは足し合わせで効き、明確に強い手がかりは単独でも効く、という折衷。

統合はすべて「警告の強さ」の空間で行う。集中のように高いほど良い軸は cue のスコアを
先に反転してから足す。スコアのまま max を取ると、良い方の cue が勝って警告が鈍るため。

平滑化は非対称にしてある。安全側の装置なので、上がるときは速く（危険を待たせない）、
下がるときは遅く（一瞬の回復で警告を解かない）。同じ係数で上下させると、警告を早くする
ほど解除も早くなり、境界付近で鳴り止み鳴り直す動きになる。
係数は判定の時刻の間隔から決める。1 フレームあたりの係数で持つと、同じ設定でも fps が
下がるほど解除が遅くなる（30fps で 0.5 秒の解除が 5fps では 3 秒になる）。
段のばたつきは LevelLatch（上げと下げで別のしきい値）が受け持つ。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ...contracts import Assessment, CueResult, Dimension, Observation
from ...geometry import clamp
from ..states import DimensionSpec, LevelLatch, alarm_of

# attack_frames / release_frames を時定数に直すときの fps。この fps で回したときに、
# フレーム数で指定した EMA（係数 2/(N+1)）と同じ動きになる時定数を使う。
REFERENCE_FPS = 30.0


class RuleBasedPolicy:
    def __init__(
        self,
        dimensions: Sequence[DimensionSpec],
        weights: dict[str, float],
        attack_frames: int = 2,
        release_frames: int = 20,
        attack_seconds: float | None = None,
        release_seconds: float | None = None,
    ) -> None:
        self._dims = tuple(dimensions)
        self._weights = dict(weights)
        self._attack_tau = _time_constant(attack_seconds, attack_frames)  # 上がるときの時定数
        self._release_tau = _time_constant(release_seconds, release_frames)  # 下がるときの時定数
        self._ema: dict[str, float] = {}
        self._last_at: float | None = None
        self._latches = {s.name: LevelLatch(s.levels, s.release_margin) for s in self._dims}

    def reset(self) -> None:
        """判定の履歴を捨てる。運転者が替わったときや再キャリブレーション時に呼ぶ。

        呼ばないと、前の人の平滑値と段がそのまま次の人の初期状態になる。
        """
        self._ema.clear()
        self._last_at = None
        for latch in self._latches.values():
            latch.reset()

    def decide(self, obs: Observation, cues: Sequence[CueResult]) -> Assessment:
        by_dim: dict[str, list[CueResult]] = {}
        for r in cues:
            by_dim.setdefault(r.dimension, []).append(r)

        dt = self._elapsed(obs.features.timestamp)
        dims: dict[str, Dimension] = {}
        for spec in self._dims:
            results = by_dim.get(spec.name, [])
            alarm = self._smooth(spec.name, self._dimension_alarm(spec, results), dt)
            score = alarm_of(spec, alarm)  # 反転は自己逆写像なので同じ関数で軸の値に戻る
            contributing = tuple(r.name for r in results if r.active)
            dims[spec.name] = Dimension(
                spec.name,
                score,
                self._latches[spec.name].update(alarm),
                contributing,
                alarm if spec.inverted else None,
                spec.alert_name,
            )
        return Assessment(dimensions=dims, timestamp=obs.features.timestamp, cues=tuple(cues))

    def _dimension_alarm(self, spec: DimensionSpec, results: Sequence[CueResult]) -> float:
        # 手がかりが1つも無い＝警告する理由が無い。反転軸でもここは 0（無言）にする。
        if not results:
            return 0.0
        if spec.combine == "weighted":
            return self._agreement_alarm(spec, [r for r in results if r.valid])
        # 平均の分母は「測れている cue」だけ。全 cue を分母に固定すると、測れない cue が
        # 0 として平均を薄める。サングラスで目の cue が4本落ちると、残る頭部の cue が
        # 満点でも平均は 3/7 にしかならず、縮退運転が成立しない。
        usable = [r for r in results if r.valid]
        weighted = 0.0
        if usable:
            weighted = self._weighted_mean([(r, alarm_of(spec, r.score)) for r in usable])
        # 単独シグナルは valid を問わない。attention_buffer のように「測れていないこと
        # 自体を根拠にして active を立てる」cue があり、そこを黙らせると顔を見失った
        # 状態が無警告になる。
        strongest = max((alarm_of(spec, r.score) for r in results if r.active), default=0.0)
        return clamp(max(weighted, strongest))

    def _agreement_alarm(self, spec: DimensionSpec, results: Sequence[CueResult]) -> float:
        """一致を要求する軸。強い順に min_agree 本を平均し、同意が足りなければ割り引く。

        分母を計測できた cue に絞るのは、値が出ない cue（rPPG が拾えない等）を 0 として
        数え続けると平均が薄まり、残る cue がどれだけ強く出ても警告に届かなくなるため。
        平均を強い min_agree 本に限るのも同じ理由で、計測できている cue が増えるほど
        黙っている cue が分母に入り、min_agree 本が揃っても満額に届かなくなる
        （4 本中 2 本が満点でも 0.5 付近に留まる）。
        min_agree が 0 なら一致を求めず、計測できている cue 全部の平均を使う。

        同意の数え方は active ではなく警告の強さで見る。単独では警告を立てない約束の
        cue（facial_tension など）は active を上げないので、active で数えると
        どれだけ揃っても同意 0 本になる。
        """
        if not results:
            return 0.0
        alarms = [(r, alarm_of(spec, r.score)) for r in results]
        if spec.min_agree <= 0:
            return clamp(self._weighted_mean(alarms))
        # 同じ強さなら重い cue を先に採る。軽い cue の 0 を選ぶと、1 本だけの兆候が
        # 重みの比で持ち上がって割り引いたあとも low を越えてしまう。
        ranked = sorted(alarms, key=lambda ra: (ra[1], self._weight(ra[0])), reverse=True)
        top = self._weighted_mean(ranked[: spec.min_agree])
        threshold = spec.levels.get("low", 0.3)
        agreeing = sum(1 for _, a in alarms if a >= threshold)
        if agreeing < spec.min_agree:
            top *= agreeing / spec.min_agree
        return clamp(top)

    def _weight(self, result: CueResult) -> float:
        return self._weights.get(result.name, 1.0)

    def _weighted_mean(self, alarms: Sequence[tuple[CueResult, float]]) -> float:
        total_w = sum(self._weight(r) for r, _ in alarms) or 1.0
        return sum(self._weight(r) * a for r, a in alarms) / total_w

    def _elapsed(self, now: float) -> float:
        # 時刻が進まない・戻ったとき（同じ時刻での再判定、入力源の張り替え）は基準 fps の
        # 1 フレームぶんとみなす。0 秒として扱うと、時刻を持たない入力で平滑値が動かなくなる。
        last, self._last_at = self._last_at, now
        if last is None or not now > last:
            return 1.0 / REFERENCE_FPS
        return now - last

    def _smooth(self, name: str, value: float, dt: float) -> float:
        prev = self._ema.get(name)
        if prev is None:
            self._ema[name] = value
            return value
        tau = self._attack_tau if value > prev else self._release_tau
        smoothed = prev + _ema_alpha(dt, tau) * (value - prev)
        self._ema[name] = smoothed
        return smoothed


def _time_constant(seconds: float | None, frames: int) -> float:
    """EMA の時定数（秒）。秒で指定が無ければ、フレーム数を REFERENCE_FPS での時定数に直す。"""
    if seconds is not None:
        return max(0.0, float(seconds))
    alpha = 2.0 / (max(1, int(frames)) + 1)
    if alpha >= 1.0:
        return 0.0
    return -1.0 / (REFERENCE_FPS * math.log(1.0 - alpha))


def _ema_alpha(dt: float, tau: float) -> float:
    return 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)
