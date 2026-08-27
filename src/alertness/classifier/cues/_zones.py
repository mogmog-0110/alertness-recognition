"""視線の行き先を車内のゾーンに分ける。

AttenD の注意バッファは「前方から目を離した時間」を数えるが、運転で目を離す先は
どれも同じではない。ミラーとメーターの確認は安全確認そのもので、これを脇見と同じ
速さで減点すると「確認を怠る運転者ほど良い点になる」という逆転が起きる。本家 AttenD も
ミラー・メーターへの一瞥には、減り始めるまでの猶予を別に与えている。

どこに何があるかは車種と取り付けで変わるので、境界は全部設定から渡す。既定値は
「カメラを注視すべき方向（前方）に置く」前提の暫定値で、実車では必ず測り直すこと。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Zone(str, Enum):
    """視線の行き先。運転として妥当な順に並べてある。"""

    FORWARD = "forward"  # 前方。ここだけが注意残高を回復させる
    INSTRUMENT = "instrument"  # メーター・車載表示。短い一瞥は正常
    MIRROR = "mirror"  # ミラー・目視確認。短い一瞥は正常
    AWAY = "away"  # それ以外。脇見


@dataclass(frozen=True)
class ZoneMap:
    """頭部の向きと視線のズレから、ゾーンの境界を決める設定。

    enabled は既定で False。ミラーとメーターの角度は車種とシート位置と取り付けで
    まったく変わるので、測らずに既定値を当てると「脇見をミラー確認と読む」ことになり、
    区別しない場合より危険になる。実車で角度を測ってから有効にすること。
    無効の間は前方かそれ以外かの2値で、従来どおりの振る舞いになる。
    """

    forward_gaze: float = 0.035  # これ以内の視線ズレなら前方
    forward_yaw: float = 12.0  # これ以内の横向きなら前方
    # 視線は水平方向しか取れない（虹彩の左右位置しか見ていない）ので、下を向いたことは
    # 頭部の pitch でしか分からない。前方の条件から pitch を外すと、膝元を見ている姿勢が
    # 「視線ズレ 0・横向き 0」で前方と判定される。
    forward_pitch: float = 10.0  # これ以内の縦の傾きなら前方
    enabled: bool = False  # ミラー／メーターのゾーンを使うか
    mirror_yaw: float = 35.0  # ここまでの横向きはミラー確認とみなす
    instrument_pitch: float = 10.0  # これ以上の下向きはメーターを見ている角度
    instrument_pitch_max: float = 22.0  # これを超える下向きは膝元＝脇見
    instrument_yaw: float = 15.0  # メーターは正面寄り。これより横ならミラー扱い

    def __post_init__(self) -> None:
        # 設定は YAML から来るので型が保証されない。enabled に数値が入ると、測っていない
        # 角度でミラー・メーターを区別し始める（区別しないより危険な側に倒れる）ので、
        # 黙って通さずここで止める。
        if not isinstance(self.enabled, bool):
            raise TypeError(
                "zones.enabled は true / false で指定してください"
                f"（受け取った値: {self.enabled!r}）。"
            )

    def classify(self, gaze_dx: float, yaw: float, pitch: float) -> Zone:
        """ゾーンを1つ返す。gaze_dx が NaN（虹彩が取れない）なら頭部の向きだけで決める。

        下向きの判定に上限を置いているのは、メーターと膝元のスマホが同じ「下向き＋
        正面寄り」になるため。上限が無いと、最も危険な脇見が最も安全なゾーンに化ける。
        """
        centered = math.isnan(gaze_dx) or abs(gaze_dx) <= self.forward_gaze
        if centered and abs(yaw) <= self.forward_yaw and abs(pitch) <= self.forward_pitch:
            return Zone.FORWARD
        if not self.enabled:
            return Zone.AWAY
        looking_down = self.instrument_pitch <= pitch <= self.instrument_pitch_max
        if looking_down and abs(yaw) <= self.instrument_yaw:
            return Zone.INSTRUMENT
        if abs(yaw) <= self.mirror_yaw and pitch < self.instrument_pitch:
            return Zone.MIRROR
        return Zone.AWAY


# ゾーンごとの「減り始めるまでの猶予」（秒）。安全確認には時間を与え、脇見には与えない。
# 猶予は一瞥の長さより短くしてある。ミラー確認は 0.6〜1.0 秒が普通なので、全部を無罰に
# すると、短い確認を繰り返して前方をほとんど見ない運転（視覚的時分割）を素通ししてしまう。
# 0.5 秒なら通常の確認は残高をほとんど削らず、繰り返せば削れて残高が戻らなくなる。
# 猶予を過ぎればどのゾーンでも同じ速さで減る（ミラーを見つめ続けるのも前方不注意）。
DEFAULT_LATENCY = {
    Zone.FORWARD: 0.0,
    Zone.INSTRUMENT: 0.4,
    Zone.MIRROR: 0.5,
    Zone.AWAY: 0.1,
}


def latency_of(zone: Zone, overrides: dict[str, float] | None) -> float:
    if overrides and zone.value in overrides:
        return float(overrides[zone.value])
    return DEFAULT_LATENCY[zone]
