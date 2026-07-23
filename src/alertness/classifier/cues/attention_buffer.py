"""集中（注意）の手がかり。AttenD 方式の注意バッファで「注意の残高」を追う。

Kircher & Ahlström の AttenD は、運転者の視線が前方から外れている時間を1本のバッファで
数える方式。対象を見ている間はバッファが満たされ、外れている間は減り、空になったら
「注意が向いていない」と判定する。容量 2 秒は「前方から 2 秒視線を外すと車線内の位置把握が
崩れる」という実験的な根拠から来ていて、しきい値に理由がある数少ない指標。

「末尾から連続で何秒見ているか」を数える素朴な方式に対する利点は、時間をまたいで積算する
こと。ちらちら視線を外して戻す（視覚的時分割）を繰り返すと残高は戻りきらず、外れが積もる。
連続時間を数える方式は、視線を戻した瞬間に満点へ復帰してしまい、この振る舞いを拾えない。

注視「対象」がどこかはこのアプリの設置に依存する（カメラは注意を向けるべき方向に置く前提）。
視線ズレと頭部の向きの両方が中立付近にあるフレームを「対象を見ている」とみなす。
"""

from __future__ import annotations

from ...contracts import CueResult, Observation
from ...geometry import clamp

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
    ) -> None:
        self.capacity_seconds = capacity_seconds  # 注意の残高の上限（AttenD の 2 秒）
        self.latency_seconds = latency_seconds  # 外れてから減り始めるまでの猶予
        self.gaze_on_threshold = gaze_on_threshold  # これ以下なら視線が対象に載っている
        self.on_target_yaw_deg = on_target_yaw_deg  # これ以下なら頭部が対象を向いている
        self.refill_rate = refill_rate  # 戻すときの速さ（1.0＝減るのと同じ速さ）
        self._buffer = capacity_seconds  # 残高（秒）。満タンから始める
        self._last_time: float | None = None
        self._away_seconds = 0.0  # 連続で外れている時間

    def evaluate(self, obs: Observation) -> CueResult:
        features = obs.features
        now = features.timestamp
        step = self._step(now)

        if not features.face_present:
            # 顔が見えない＝対象を見ている確証が無い。外れ扱いで減らす。
            self._away_seconds += step
            self._drain(step)
            return self._result("顔なし", valid=False)

        gaze_off = features.get("gaze_off", 1.0)
        yaw = features.get("yaw_rel", 90.0)
        on_target = gaze_off <= self.gaze_on_threshold and abs(yaw) <= self.on_target_yaw_deg

        if on_target:
            self._away_seconds = 0.0
            self._buffer = min(self.capacity_seconds, self._buffer + step * self.refill_rate)
            detail = f"注意残高 {self._buffer:.1f}/{self.capacity_seconds:.0f}s"
        else:
            self._away_seconds += step
            self._drain(step)
            detail = f"視線外れ {self._away_seconds:.1f}s 残高 {self._buffer:.1f}s"
        return self._result(detail)

    def _step(self, now: float) -> float:
        last = self._last_time
        self._last_time = now
        if last is None or now <= last:
            return 0.0
        return min(now - last, _MAX_STEP)

    def _drain(self, step: float) -> None:
        # 外れた直後の latency のぶんは減らさない（周辺視で拾える猶予）。
        drained = min(step, max(0.0, self._away_seconds - self.latency_seconds))
        self._buffer = max(0.0, self._buffer - drained)

    def _result(self, detail: str, valid: bool = True) -> CueResult:
        score = clamp(self._buffer / self.capacity_seconds) if self.capacity_seconds > 0 else 0.0
        # active＝警告に効いている状態。この軸は低いほど警告なので「残高が尽きた」ときに立てる。
        return CueResult(self.name, self.dimension, score, self._buffer <= 0.0, detail, None, valid)
