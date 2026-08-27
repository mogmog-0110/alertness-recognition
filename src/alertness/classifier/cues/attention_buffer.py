"""集中（注意）の手がかり。AttenD 方式の注意バッファで「注意の残高」を追う。

Kircher & Ahlström の AttenD は、運転者の視線が前方から外れている時間を1本のバッファで
数える方式。対象を見ている間はバッファが満たされ、外れている間は減り、空になったら
「注意が向いていない」と判定する。容量 2 秒は「前方から 2 秒視線を外すと車線内の位置把握が
崩れる」という実験的な根拠から来ていて、しきい値に理由がある数少ない指標。

「末尾から連続で何秒見ているか」を数える素朴な方式に対する利点は、時間をまたいで積算する
こと。ちらちら視線を外して戻す（視覚的時分割）を繰り返すと残高は戻りきらず、外れが積もる。
連続時間を数える方式は、視線を戻した瞬間に満点へ復帰してしまい、この振る舞いを拾えない。

外した先は区別する（_zones.py）。ミラーとメーターの確認は安全確認そのものなので、
減り始めるまでの猶予を長くとる。区別しないと、確認を怠る運転者ほど高得点になる。
猶予を過ぎればどのゾーンでも同じ速さで減る（ミラーを見つめ続けるのも前方不注意）。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...contracts import CueResult, Observation
from ...geometry import clamp
from ._forward import ForwardPose
from ._zones import Zone, ZoneMap, latency_of

_MAX_STEP = 0.5  # フレーム間隔の上限（秒）。取りこぼしで一気に減らさないための保険。


class AttentionBufferCue:
    name = "attention_buffer"
    dimension = "concentration"

    def __init__(
        self,
        capacity_seconds: float = 2.0,
        latency_seconds: float = 0.1,
        gaze_on_threshold: float = 0.035,
        on_target_yaw_deg: float = 25.0,
        refill_rate: float = 1.0,
        zone_latency: dict[str, float] | None = None,
        zones: Mapping[str, Any] | None = None,
        auto_forward: bool = False,
        forward_seconds: float = 180.0,
        forward_min_samples: int = 60,
    ) -> None:
        self.capacity_seconds = capacity_seconds  # 注意の残高の上限（AttenD の 2 秒）
        self.refill_rate = refill_rate  # 戻すときの速さ（1.0＝減るのと同じ速さ）
        # 前方の定義は cue のしきい値から作る。ゾーンを使わない設置（机上のデモなど）でも
        # 従来どおり「視線ズレと頭部の向きが中立付近＝対象を見ている」が保たれる。
        self._zones = ZoneMap(
            forward_gaze=gaze_on_threshold,
            forward_yaw=on_target_yaw_deg,
            **(zones or {}),
        )
        # 明示指定が無いゾーンは _zones.DEFAULT_LATENCY を使う。AWAY だけは従来の
        # latency_seconds を引き継ぐ（既存の設定を書き換えずに挙動を保てるように）。
        self._latency = {Zone.AWAY.value: latency_seconds, **(zone_latency or {})}
        # 前方の向きを走行中の分布から推定する。起動時キャリブのずれを引き継がずに済む。
        self._forward = ForwardPose(forward_seconds, forward_min_samples) if auto_forward else None
        self._buffer = capacity_seconds  # 残高（秒）。満タンから始める
        self._last_time: float | None = None
        self._away_seconds = 0.0  # 連続で前方から外れている時間
        self._zone = Zone.FORWARD

    def reset(self) -> None:
        """残高を満タンに戻す。運転者が替わったとき、前の人の外れ具合を持ち越さない。"""
        self._buffer = self.capacity_seconds
        self._last_time = None
        self._away_seconds = 0.0
        self._zone = Zone.FORWARD
        if self._forward is not None:
            self._forward.reset()

    def evaluate(self, obs: Observation) -> CueResult:
        features = obs.features
        now = features.timestamp
        step = self._step(now)

        if not features.face_present:
            # 顔が見えない＝前方を見ている確証が無い。外れ扱いで減らす。
            self._zone = Zone.AWAY
            self._away_seconds += step
            self._drain(step)
            return self._result("顔なし", valid=False)

        gaze_dx = features.get("gaze_dx", float("nan"))
        yaw = features.get("yaw_rel", 90.0)
        pitch = features.get("pitch_rel", 0.0)
        if self._forward is not None:
            # 積むのは補正前の生の値。補正後を積むと、いま前方だと思っている向きの
            # 標本だけが集まり、最初のずれがそのまま固定される。
            self._forward.update(now, gaze_dx, yaw, pitch)
            gaze_dx, yaw, pitch = self._forward.correct(gaze_dx, yaw, pitch)
        self._zone = self._zones.classify(gaze_dx, yaw, pitch)
        if self._zone is Zone.FORWARD:
            self._away_seconds = 0.0
            self._buffer = min(self.capacity_seconds, self._buffer + step * self.refill_rate)
            return self._result(f"注意残高 {self._buffer:.1f}/{self.capacity_seconds:.0f}s")

        self._away_seconds += step
        self._drain(step)
        detail = f"{_LABELS[self._zone]} {self._away_seconds:.1f}s 残高 {self._buffer:.1f}s"
        return self._result(detail)

    def _step(self, now: float) -> float:
        last = self._last_time
        self._last_time = now
        if last is None or now <= last:
            return 0.0
        return min(now - last, _MAX_STEP)

    def _drain(self, step: float) -> None:
        # 外れた直後の猶予ぶんは減らさない。長さは行き先で変える（ミラー確認は無罰の一瞥）。
        grace = latency_of(self._zone, self._latency)
        drained = min(step, max(0.0, self._away_seconds - grace))
        self._buffer = max(0.0, self._buffer - drained)

    def _result(self, detail: str, valid: bool = True) -> CueResult:
        score = clamp(self._buffer / self.capacity_seconds) if self.capacity_seconds > 0 else 0.0
        progress = self._forward.progress() if self._forward is not None else None
        # active＝警告に効いている状態。この軸は低いほど警告なので「残高が尽きた」ときに立てる。
        return CueResult(
            self.name, self.dimension, score, self._buffer <= 0.0, detail, progress, valid
        )


_LABELS = {
    Zone.FORWARD: "前方",
    Zone.INSTRUMENT: "メーター",
    Zone.MIRROR: "ミラー",
    Zone.AWAY: "視線外れ",
}
