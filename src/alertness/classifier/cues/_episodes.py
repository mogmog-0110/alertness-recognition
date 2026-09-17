"""目の開き具合の時系列から閉眼エピソードを切り出す共通部品。

瞬きの長さ・まぶたの戻りの遅さ・瞬きの頻度は、どれも「1回の閉眼」を単位に数える。
切り出しをここに一本化して、各 cue は数え方だけを持つ。

閉眼の入口と出口で別のしきい値を使う（ヒステリシス）。1本のしきい値で切ると、
まばたきの途中で値が境界をまたいで震えるたびに1回の閉眼が2回3回に割れ、
「短い瞬きが多発している」という実際とは逆の像になる。

時刻はしきい値をまたいだ 2 フレームの間を線形補間して出す。端末経由では 15〜25fps
しか出ず、フレーム単位に丸めると 1 回の瞬きの長さに 40〜70ms の段差が乗る。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# 戻りは「瞬き直前の開き具合までの半分」に届いた時刻で測る。ランドマークの値は開いた後に
# 数百 ms かけて元の値へ漸近する尾を引くので、9 割のような高い目標では尾の長さを測ってしまう。
# 半分なら、まぶたが持ち上がる速い区間だけで決まる。
REOPEN_FRACTION = 0.5
# 瞬き直前の開き具合を探す長さ（秒）。この区間の最大値を「開いていた位置」とみなす。
_LOOKBACK_SECONDS = 1.0


@dataclass(frozen=True)
class Closure:
    """1回の閉眼。時刻はすべて秒。"""

    start: float  # 閉じ始め（閉眼しきい値を下回った時刻）
    end: float  # 開き始め（開眼しきい値を上回った時刻）
    bottom: float  # 最も閉じた時刻
    reopened: float | None  # 瞬き直前の開き具合の半分まで戻った時刻。測れなければ None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def reopen_seconds(self) -> float | None:
        """最も閉じた瞬間から半分まで戻るまでの時間。

        眠気が進むとまぶたを持ち上げる動きが遅くなるので、閉眼の長さとは別に伸びる。
        測れなかったものは None（0 と混同しない）。
        """
        if self.reopened is None:
            return None
        return max(0.0, self.reopened - self.bottom)


def closure_episodes(
    times: Sequence[float],
    ears: Sequence[float],
    closed_ratio: float,
    open_ratio: float,
) -> list[Closure]:
    """閉眼エピソードを古い順に返す。

    ears は開眼基準で正規化した開き具合（1.0 が楽に開けた状態）。closed_ratio を
    下回ったら閉じ始め、open_ratio を上回ったら開き始めとみなす。
    open_ratio は closed_ratio より大きいこと（同値なら震えで分割される）。
    末尾がまだ閉じたままのエピソードは、進行中なので返さない。
    """
    episodes: list[Closure] = []
    start: float | None = None
    start_index = 0
    bottom_index = 0

    for i, ear in enumerate(ears):
        if start is None:
            if ear < closed_ratio:
                start = _crossing(times, ears, i, closed_ratio)
                start_index = bottom_index = i
            continue
        if ear < ears[bottom_index]:
            bottom_index = i
        if ear >= open_ratio:
            end = _crossing(times, ears, i, open_ratio)
            reopened = _reopen_time(times, ears, start_index, bottom_index, closed_ratio)
            episodes.append(Closure(start, end, times[bottom_index], reopened))
            start = None
    return episodes


def _reopen_time(
    times: Sequence[float],
    ears: Sequence[float],
    start_index: int,
    bottom_index: int,
    closed_ratio: float,
) -> float | None:
    """最も閉じた位置から、瞬き直前の開き具合までの REOPEN_FRACTION に戻った時刻。

    探すのは次の閉眼が始まるまで。その先まで探すと、戻りきらなかった瞬きに次の瞬きの
    向こう側の時刻が付き、瞬きの間隔をまぶたの戻りの遅さと読んでしまう。
    目標は校正時の開眼基準ではなく瞬き直前の値から決める。普段の開き具合が基準より
    細い人でも、同じ物差しで測れる。
    """
    before = min(_open_level(times, ears, start_index), 1.0)
    bottom = ears[bottom_index]
    goal = bottom + REOPEN_FRACTION * (before - bottom)
    reopening = False
    for i in range(bottom_index + 1, len(ears)):
        if ears[i] >= goal:
            return _crossing(times, ears, i, goal)
        reopening = reopening or ears[i] > bottom
        if reopening and ears[i] < closed_ratio <= ears[i - 1]:
            return None  # 戻りきる前に次の瞬きに入った
    return None


def _open_level(times: Sequence[float], ears: Sequence[float], index: int) -> float:
    """index の直前 _LOOKBACK_SECONDS の最大値。前が無ければ開眼基準の 1.0。"""
    earliest = times[index] - _LOOKBACK_SECONDS
    prior = [ears[k] for k in range(index) if times[k] >= earliest]
    return max(prior) if prior else 1.0


def _crossing(times: Sequence[float], ears: Sequence[float], index: int, level: float) -> float:
    """index-1 と index の間で level をまたいだ時刻。前のフレームが無ければ index の時刻。"""
    if index == 0:
        return times[0]
    a, b = ears[index - 1], ears[index]
    if a == b:
        return times[index]
    ratio = min(1.0, max(0.0, (level - a) / (b - a)))
    return times[index - 1] + ratio * (times[index] - times[index - 1])
