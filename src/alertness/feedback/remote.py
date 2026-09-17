"""判定結果を iPhone へ返す出力先。映像を送ってきたのと同じ接続を使う。

端末は運転者の手元にあるので、ここが実際の警告の出口になる。画面の文言と、音・振動を
出すかどうかだけを送る。特徴量は開発中の切り分け用で、無くても端末は動く。

鳴らす時点はこちらで決めて beep として送る。端末に間隔を持たせると、設定の
alert_cooldown_seconds や HIGH での間隔の詰めが効かず、PC の窓と鳴り方が食い違う。
sounds に無い軸（stress）は表示だけにする。緊張している運転者に警告音を重ねない。

軸の表示名は config から受け取る。OpenCV の窓は日本語を描けないので軸名は英語のままに
してあるが、端末の画面は日本語で出せる。名前の対応をここに埋め込むと軸を増やすたびに
コードを触ることになるので、対応表は設定に置く。
"""

from __future__ import annotations

from collections.abc import Mapping

from ..contracts import Assessment, Dimension, Level, Observation
from .cadence import AlertCadence

_LEVEL_NAME = {
    Level.NONE: "none",
    Level.LOW: "low",
    Level.MEDIUM: "medium",
    Level.HIGH: "high",
}


class RemoteSink:
    def __init__(
        self,
        link,
        alert_from: Level = Level.MEDIUM,
        features: tuple[str, ...] = (),
        names: Mapping[str, str] | None = None,
        sounds: Mapping[str, str] | None = None,
        cadence: AlertCadence | None = None,
    ) -> None:
        self._link = link
        self._alert_from = alert_from
        self._features = features
        self._names = dict(names or {})
        self._sounds = dict(sounds or {})
        self._cadence = cadence or AlertCadence()
        self._last_beep: float | None = None
        # ガイド中に判定 payload も送ると、端末が指示と判定を毎フレーム
        # 行き来して激しく点滅する (実測: 30fps でそのまま切り替わった)。
        # 指示が来ている間は画面を指示に譲る。
        self._guiding_until = 0.0

    def emit(self, obs: Observation, assessment: Assessment) -> None:
        if obs.features.timestamp < self._guiding_until:
            return  # 収録中。画面は指示が持っている
        level = assessment.alert_level()
        payload: dict = {
            "timestamp": assessment.timestamp,
            "phase": "running",
            "level": _LEVEL_NAME.get(level, "none"),
            "alert": level >= self._alert_from,
        }
        head = assessment.headline()
        if head is not None:
            # 対応表の見出しは画面に出る名前（alert_name があればそちら）。集中の軸は
            # 「集中」ではなく「注意散漫」として警告するので、警告名の側で引く。
            # alert_name は設定の対応表にあるときだけ意味キーとして扱う。日本語などの
            # 表示名を直接持つ既存 Dimension では、安定した内部名 name をキーにする。
            dimension_key = head.alert_name if head.alert_name in self._names else head.name
            shown = self._names.get(head.display_name, head.display_name)
            payload["dimension_key"] = dimension_key
            payload["dimension"] = shown
            payload["message"] = self._message(shown, level)
            cause = _cause(head, assessment)
            if cause:
                payload["cause"] = cause
        beep = self._beep(assessment, obs.features.timestamp)
        if beep is not None:
            payload["beep"] = beep
        if self._features:
            payload["features"] = self._selected(obs)
        self._link.send(payload)

    def guiding(
        self, obs: Observation, title: str, instruction: str,
        phase: str, remaining: float, progress: float, prompt_key: str = "",
    ) -> None:
        """収録の指示を端末へ送る。運転者は PC の窓を見られない。

        ready は「次はこれをやる」の予告、hold が実際の記録区間。端末側は
        残り秒数と全体の進捗を出す。
        """
        # 指示が途切れるまでは判定を送らない。両方を毎フレーム送ると、端末が
        # 指示と判定を 30 回/秒で行き来して激しく点滅する。
        self._guiding_until = obs.features.timestamp + 1.0
        self._link.send(
            {
                "timestamp": obs.features.timestamp,
                "phase": "guided",
                "guided": {
                    "title": title,
                    "instruction": instruction,
                    "step": phase,
                    "remaining": round(float(remaining), 1),
                    "progress": max(0.0, min(1.0, float(progress))),
                    "prompt_key": prompt_key,
                },
                "alert": False,
            }
        )

    def preparing(self, obs: Observation) -> None:
        """端末が開き直してから測り始めるまでの間。判定は送らない。"""
        self._restart_cadence()
        self._link.send(
            {"timestamp": obs.features.timestamp, "phase": "preparing", "alert": False}
        )

    def calibrating(
        self, obs: Observation, progress: float,
        waiting_for: str = "", expected_seconds: float = 0.0,
    ) -> None:
        # 端末はこれを見て「正面を見てください」と進捗を出し、警告を鳴らさない。
        # 基準が無いままの判定で鳴らしても意味が無い。
        #
        # waiting_for は「進捗が伸びない理由」。心拍の基準は 20 秒の窓が満ちる
        # までゼロのままなので、理由を出さないと故障に見える。
        # progress は素の値をそのまま送る。心拍待ちの間は 0 のまま動かず最後に
        # 跳ねる量なので、端末はバーではなく回転表示で「動いている」ことだけを
        # 伝える。経過時間で滑らかに見せるのは、実態と違う値を出すことになる。
        self._restart_cadence()  # 測り終えた直後の警告は、前の続きではなく最初の 1 回
        payload = {
            "timestamp": obs.features.timestamp,
            "phase": "calibrating",
            "progress": max(0.0, min(1.0, float(progress))),
            "message": "基準を測っています",
            "alert": False,
        }
        if waiting_for:
            payload["waiting_for"] = waiting_for
        self._link.send(payload)

    def _restart_cadence(self) -> None:
        self._cadence.reset()
        self._last_beep = None

    def _beep(self, assessment: Assessment, now: float) -> dict[str, str] | None:
        """いま鳴らす音。複数の軸が同時に鳴らし時なら段の高い方を 1 つだけ。

        鳴らさない軸も含めて全軸を毎回 cadence に通す。収まったことが伝わらないと、
        次に立ったときに詰めた間隔から鳴り始める。選ばれなかった軸は鳴らしたことに
        しない。ただし直後に続けて鳴らすと 2 つの音が重なるので、どの軸でも前の音から
        min_interval は空ける。
        """
        pending = [
            dim
            for dim in assessment.dimensions.values()
            if dim.name in self._sounds and self._cadence.pending(dim.name, dim.level, now)
        ]
        recent = self._last_beep is not None and now - self._last_beep < self._cadence.min_interval
        if not pending or recent:
            return None
        chosen = max(pending, key=lambda d: d.level)
        self._cadence.mark(chosen.name, chosen.level, now)
        self._last_beep = now
        return {"sound": self._sounds[chosen.name], "level": _LEVEL_NAME[chosen.level]}

    def _selected(self, obs: Observation) -> dict[str, float]:
        # NaN は JSON では表せない（json は NaN を吐くが受け側の JSONDecoder が拒む）。
        # 測れていない特徴はキーごと落とし、端末側では「欠けている」と扱わせる。
        return {
            name: value
            for name, value in obs.features.values.items()
            if name in self._features and value == value
        }

    @staticmethod
    def _message(name: str, level: Level) -> str:
        if level >= Level.HIGH:
            return f"{name}が強いです"
        if level == Level.MEDIUM:
            return f"{name}の兆候があります"
        return name

    def close(self) -> None:
        pass  # 接続は RemoteLink の持ち物。source 側が閉じる


def _cause(head: Dimension, assessment: Assessment) -> str:
    """見出しの軸で一番強く効いている cue の名前。無ければ空文字。

    軸の名前だけだと、顔を見失ったときも「眠気が強い」と出て運転者が戸惑う。
    端末は cue 名から「顔が映っていません」のような原因の文に引き直す。
    向きが反転する軸（集中）は score が低いほど警告なので、最小を取る。
    """
    active = set(head.contributing)
    candidates = [c for c in assessment.cues if c.name in active and c.dimension == head.name]
    if not candidates:
        return ""
    pick = min if head.alert_score is not None else max
    return pick(candidates, key=lambda c: c.score).name
