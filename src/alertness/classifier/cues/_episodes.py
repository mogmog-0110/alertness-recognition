"""EAR の時系列から閉眼エピソードを切り出す共通部品。

瞬きの長さ・まぶたの戻りの遅さ・瞬きの頻度は、どれも「1回の閉眼」を単位に数える。
切り出しをここに一本化して、各 cue は数え方だけを持つ。

閉眼の入口と出口で別のしきい値を使う（ヒステリシス）。1本のしきい値で切ると、
まばたきの途中で EAR が境界をまたいで震えるたびに1回の閉眼が2回3回に割れ、
「短い瞬きが多発している」という実際とは逆の像になる。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Closure:
    """1回の閉眼。時刻はすべて秒。"""

    start: float  # 閉じ始め（閉眼しきい値を下回った時刻）
    end: float  # 開き始め（開眼しきい値を上回った時刻）
    bottom: float  # 最も閉じた時刻
    reopened: float | None  # 開眼基準まで戻った時刻。窓の途中で切れていれば None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def reopen_seconds(self) -> float | None:
        """最も閉じた瞬間から開眼基準へ戻るまでの時間。

        眠気が進むとまぶたを持ち上げる動きが遅くなるので、閉眼の長さとは別に伸びる。
        戻りきる前に窓が終わっていれば None（測れていない、を 0 と混同しない）。
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

    ears は開眼基準で正規化済みの EAR（1.0 が楽に開けた状態）。closed_ratio を
    下回ったら閉じ始め、open_ratio を上回ったら開き始めとみなす。
    open_ratio は closed_ratio より大きいこと（同値なら震えで分割される）。
    末尾がまだ閉じたままのエピソードは、進行中なので返さない。
    """
    episodes: list[Closure] = []
    start: float | None = None
    bottom_t = 0.0
    bottom_v = float("inf")

    for i, (t, ear) in enumerate(zip(times, ears, strict=True)):
        if start is None:
            if ear < closed_ratio:
                start, bottom_t, bottom_v = t, t, ear
            continue
        if ear < bottom_v:
            bottom_t, bottom_v = t, ear
        if ear >= open_ratio:
            episodes.append(Closure(start, t, bottom_t, _reopen_time(times, ears, i)))
            start = None
            bottom_v = float("inf")
    return episodes


def _reopen_time(times: Sequence[float], ears: Sequence[float], from_index: int) -> float | None:
    """開き始めた地点から、開眼基準(1.0)の 9 割まで戻った最初の時刻。

    9 割で見るのは、正規化した EAR がぴったり 1.0 に戻るとは限らないため。
    戻りきらないまま窓が終わったら None。
    """
    for i in range(from_index, len(ears)):
        if ears[i] >= 0.9:
            return times[i]
    return None
