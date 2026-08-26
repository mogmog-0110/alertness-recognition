"""目の信号が使える状態かを見る。サングラス・暗所・逆光での縮退運転の判断。

サングラスをかけても顔検出は成功し、目のランドマークも「それらしい」位置に出る。
値は出ているのに中身が無い状態なので、顔の有無では弾けない。しかも EAR が低めに
張り付くと PERCLOS が上がり、**目を閉じていないのに眠気を警告する**という、最も
避けたい向きの誤りになる。暗所や強い逆光でも同じことが起きる。

判断の根拠は瞬きの有無。人は安静時でも 10〜20 回/分は瞬きをするので、十分な長さの窓に
1回も瞬きが無ければ、閉眼の値がまぶたの動きを表していない。値そのものの妥当性ではなく
「動いているか」で見るのは、サングラス越しの EAR が低いまま安定するという壊れ方を
そのまま捉えられるため。

使えないと分かったら、目に依存する cue が黙る。眠気の判定は頭部（うなずき・うつむき）と
あくびと見失いで続く。精度は落ちるが、黙るよりはよい。
"""

from __future__ import annotations

from ...contracts import Observation
from ._episodes import closure_episodes
from ._support import window_values


def eye_signal_usable(
    obs: Observation,
    window_seconds: float = 60.0,
    min_blinks: int = 1,
    closed_ratio: float = 0.6,
    open_ratio: float = 0.7,
) -> tuple[bool, str]:
    """(使えるか, 使えない理由) を返す。

    窓が埋まりきっていない間は「使える」を返す。起動直後をサングラスと読むと、
    最も判定が要る立ち上がりの数十秒が丸ごと縮退運転になる。
    """
    times, ears = window_values(obs, "ear_norm", window_seconds, 1.0)
    if len(times) < 2:
        return True, ""
    span = times[-1] - times[0]
    if span < window_seconds * 0.8:
        return True, ""

    episodes = closure_episodes(times, ears, closed_ratio, open_ratio)
    if len(episodes) >= min_blinks:
        return True, ""
    return False, f"目の信号なし（{span:.0f}秒間 瞬きを検出できず）"
