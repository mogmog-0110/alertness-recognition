"""cue の結果を評価軸ごとに統合するルールベースの方針。

軸ごとに「重み付き平均」と「最も強い単独シグナル」の大きい方を取る。
弱い手がかりは足し合わせで効き、明確に強い手がかりは単独でも効く、という折衷。

統合はすべて「警告の強さ」の空間で行う。集中のように高いほど良い軸は cue のスコアを
先に反転してから足す。スコアのまま max を取ると、良い方の cue が勝って警告が鈍るため。

平滑化は非対称にしてある。安全側の装置なので、上がるときは速く（危険を待たせない）、
下がるときは遅く（一瞬の回復で警告を解かない）。同じ係数で上下させると、警告を早くする
ほど解除も早くなり、境界付近で鳴り止み鳴り直す動きになる。
段のばたつきは LevelLatch（上げと下げで別のしきい値）が受け持つ。
"""

from __future__ import annotations

from collections.abc import Sequence

from ...contracts import Assessment, CueResult, Dimension, Observation
from ...geometry import clamp
from ..states import DimensionSpec, LevelLatch, alarm_of


class RuleBasedPolicy:
    def __init__(
        self,
        dimensions: Sequence[DimensionSpec],
        weights: dict[str, float],
        attack_frames: int = 2,
        release_frames: int = 20,
    ) -> None:
        self._dims = tuple(dimensions)
        self._weights = dict(weights)
        self._attack = _ema_alpha(attack_frames)  # 上がるときの EMA 係数
        self._release = _ema_alpha(release_frames)  # 下がるときの EMA 係数
        self._ema: dict[str, float] = {}
        self._latches = {s.name: LevelLatch(s.levels, s.release_margin) for s in self._dims}

    def reset(self) -> None:
        """判定の履歴を捨てる。運転者が替わったときや再キャリブレーション時に呼ぶ。

        呼ばないと、前の人の平滑値と段がそのまま次の人の初期状態になる。
        """
        self._ema.clear()
        for latch in self._latches.values():
            latch.reset()

    def decide(self, obs: Observation, cues: Sequence[CueResult]) -> Assessment:
        by_dim: dict[str, list[CueResult]] = {}
        for r in cues:
            by_dim.setdefault(r.dimension, []).append(r)

        dims: dict[str, Dimension] = {}
        for spec in self._dims:
            results = by_dim.get(spec.name, [])
            alarm = self._smooth(spec.name, self._dimension_alarm(spec, results))
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
            total_w = sum(self._weights.get(r.name, 1.0) for r in usable) or 1.0
            weighted = (
                sum(self._weights.get(r.name, 1.0) * alarm_of(spec, r.score) for r in usable)
                / total_w
            )
        # 単独シグナルは valid を問わない。attention_buffer のように「測れていないこと
        # 自体を根拠にして active を立てる」cue があり、そこを黙らせると顔を見失った
        # 状態が無警告になる。
        strongest = max((alarm_of(spec, r.score) for r in results if r.active), default=0.0)
        return clamp(max(weighted, strongest))

    def _agreement_alarm(self, spec: DimensionSpec, results: Sequence[CueResult]) -> float:
        """一致を要求する軸。計測できている cue だけで平均し、同意が足りなければ割り引く。

        分母を計測できた cue に絞るのは、値が出ない cue（rPPG が拾えない等）を 0 として
        数え続けると平均が薄まり、残る cue がどれだけ強く出ても警告に届かなくなるため。
        そのぶん「何本が兆候ありと言っているか」を min_agree で明示的に要求する。

        同意の数え方は active ではなく警告の強さで見る。単独では警告を立てない約束の
        cue（facial_tension など）は active を上げないので、active で数えると
        どれだけ揃っても同意 0 本になる。
        """
        if not results:
            return 0.0
        alarms = [(r, alarm_of(spec, r.score)) for r in results]
        total_w = sum(self._weights.get(r.name, 1.0) for r, _ in alarms) or 1.0
        weighted = sum(self._weights.get(r.name, 1.0) * a for r, a in alarms) / total_w
        threshold = spec.levels.get("low", 0.3)
        agreeing = sum(1 for _, a in alarms if a >= threshold)
        if spec.min_agree > 0 and agreeing < spec.min_agree:
            weighted *= agreeing / spec.min_agree
        return clamp(weighted)

    def _smooth(self, name: str, value: float) -> float:
        prev = self._ema.get(name)
        if prev is None:
            self._ema[name] = value
            return value
        alpha = self._attack if value > prev else self._release
        smoothed = alpha * value + (1.0 - alpha) * prev
        self._ema[name] = smoothed
        return smoothed


def _ema_alpha(frames: int) -> float:
    return 2.0 / (max(1, frames) + 1)
