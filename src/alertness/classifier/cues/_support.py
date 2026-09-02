"""cue 実装で共通して使う、時系列の取り出しヘルパ。

割合はフレーム数ではなく時間で数える。取り込みの fps は揺れるうえ、顔検出が落ちた
フレームは window_values が黙って除くため、フレーム数の比は「その割合が何秒ぶんか」を
表さない。30 秒窓に 2 秒ぶんしかフレームが届いていなくても、フレーム比なら満点の
PERCLOS が出てしまう。窓がどれだけ埋まっているかは window_coverage で別に見る。
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import median

from ...contracts import Observation
from ...geometry import clamp

# 標本の間隔がここまで開いたら、その空白は代表時間に数えない。標本間隔の中央値の何倍か。
# 顔なしフレームは除かれているので、空白は「見えていなかった時間」であり、直前の値が
# その間も続いていた保証はない。頭打ちにしないと、顔を見失った 10 秒がその直前の
# 1 標本の重みとして丸ごと計上される。
_MAX_GAP_FACTOR = 2.0


def window_values(
    obs: Observation, key: str, seconds: float, default: float
) -> tuple[list[float], list[float]]:
    """直近 seconds 秒ぶんの (時刻リスト, 値リスト) を返す。顔なしフレームは除く。

    顔なしを除くぶん、返る時刻は等間隔とは限らない。割合を出すときは要素数ではなく
    time_fraction を使うこと。窓がどれだけ埋まっているかは window_coverage で見る。
    """
    frames = [f for f in obs.history.recent(seconds) if f.face_present]
    times = [f.timestamp for f in frames]
    values = [f.get(key, default) for f in frames]
    return times, values


def sample_durations(times: Sequence[float]) -> list[float]:
    """各標本が代表する時間（秒）。

    標本 i は次の標本までの間を代表し、末尾は次が無いので直前の間隔で代用する。
    大きく空いた間隔は間隔中央値の _MAX_GAP_FACTOR 倍で頭打ちにする。
    標本が 1 点だけなら 1.0 を返す（割合の分母分子で相殺するので値自体に意味はない）。
    """
    n = len(times)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    gaps = [times[i + 1] - times[i] for i in range(n - 1)]
    cap = _MAX_GAP_FACTOR * median(gaps)
    capped = [min(g, cap) if cap > 0 else g for g in gaps]
    return capped + [capped[-1]]


def time_fraction(times: Sequence[float], flags: Sequence[bool]) -> float:
    """True が占める時間の割合（0..1）。フレーム数ではなく時間で数える。"""
    durations = sample_durations(times)
    if not durations:
        return 0.0
    total = sum(durations)
    if total <= 0:
        return 0.0
    return sum(d for d, f in zip(durations, flags, strict=True) if f) / total


def window_coverage(obs: Observation, seconds: float) -> float:
    """窓のうち顔が見えていた時間の割合（0..1）。

    分母は窓の公称長ではなく、履歴に実際にある時間幅。公称長で割ると、起動直後の
    まだ窓が埋まっていない時間帯にカバレッジ不足と判定され、全 cue が黙ってしまう。
    """
    frames = obs.history.recent(seconds)
    if len(frames) < 2:
        return 0.0
    span = frames[-1].timestamp - frames[0].timestamp
    if span <= 0:
        return 0.0
    present = [f.timestamp for f in frames if f.face_present]
    if len(present) < 2:
        return 0.0
    return clamp(sum(sample_durations(present)) / span)


def trailing_true_seconds(times: Sequence[float], flags: Sequence[bool]) -> float:
    """末尾から連続して True が続いている時間（秒）。"""
    if not times or not flags or not flags[-1]:
        return 0.0
    start = times[-1]
    for t, c in zip(reversed(times), reversed(flags), strict=True):
        if not c:
            break
        start = t
    return max(0.0, times[-1] - start)


def recency_weight(age_seconds: float, half_life_seconds: float) -> float:
    """古い証拠ほど軽くするための重み。半減期で指数的に減る。

    箱型の窓は「窓に入っている間は同じ重み、出たら消える」ので、姿勢を直しても
    最大で窓の長さぶん警告が残る（実測: 眠気の警告が最長 30 秒続いた）。運転者が
    直したのに鳴り続ける警告は、警告として働かない。

    かといって窓を縮めると、ゆっくり繰り返す兆候を取り逃がす。古い証拠を残しつつ
    重みだけ落とすことで、貯める性質と戻りの速さを両立させる。
    """
    if half_life_seconds <= 0:
        return 1.0
    return float(0.5 ** (max(0.0, age_seconds) / half_life_seconds))


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """重み付き平均。重みの合計が 0 なら素の平均に落とす。"""
    total = sum(weights)
    if total <= 0:
        return sum(values) / len(values) if values else 0.0
    return sum(v * w for v, w in zip(values, weights, strict=True)) / total


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """重み付き中央値。重みの合計が 0 なら素の中央値に落とす。

    平均は外れ値 1 個で跳ねる。瞬きは 60 秒の窓に数回しか入らないので、
    1 回の長い閉眼 (検出の途切れや、意図的に目を閉じた瞬間) がそのまま判定に
    なる。実測では中央値 100ms なのに平均が 422ms まで上がり、眠気の警告が
    立ち続けた。
    """
    pairs = sorted(zip(values, weights, strict=True))
    total = sum(w for _, w in pairs)
    if not pairs:
        return 0.0
    if total <= 0:
        middle = len(pairs) // 2
        if len(pairs) % 2:
            return pairs[middle][0]
        return (pairs[middle - 1][0] + pairs[middle][0]) / 2
    seen = 0.0
    for value, weight in pairs:
        seen += weight
        if seen >= total / 2:
            return value
    return pairs[-1][0]
