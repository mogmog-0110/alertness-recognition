"""イベント単位の採点。フレーム単位の accuracy では見えないことを測る。

車載で効く数字は2つしかない:
- 危険が始まってから何秒で警告できたか（検出遅延）
- 平常時に何回よけいに鳴ったか（時間あたりの誤警告）

フレーム単位の accuracy と macro-F1 は、この2つのどちらも表さない。収録の大半は平常
区間なので、警告を一切出さない判定器でも accuracy は高く出る。逆に 1 回のエピソードを
1 秒遅れで確実に捕まえる判定器と、10 秒遅れで捕まえる判定器の差もほとんど出ない。

そして実運用で最初に起きる失敗は「誤警告が多くて運転者が装置を切る」ことなので、
時間あたりの誤警告回数を第一指標に置く。切られた装置の検出率は 0 になる。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# 同じ警告が細かく切れて鳴り直したものを別々に数えないための間隔（秒）。
# これより短い無警告のすき間は、1回の警告が途切れたものとして繋ぐ。
_MERGE_GAP = 2.0


@dataclass(frozen=True)
class Episode:
    """時間の区間。正解の危険区間にも、判定が出した警告区間にも使う。"""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlaps(self, other: Episode) -> bool:
        return self.start < other.end and other.start < self.end


def episodes_from_flags(times: Sequence[float], flags: Sequence[bool]) -> list[Episode]:
    """True が続く区間を取り出す。近すぎる区間は 1 つに繋ぐ。

    繋がないと、境界付近で 1 回の警告が数回に割れ、誤警告の回数が実際より多く見える。
    """
    raw: list[Episode] = []
    start: float | None = None
    previous = times[0] if times else 0.0
    for t, flag in zip(times, flags, strict=True):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            raw.append(Episode(start, previous))
            start = None
        previous = t
    if start is not None:
        raw.append(Episode(start, previous))
    return _merge(raw)


def _merge(episodes: Sequence[Episode]) -> list[Episode]:
    merged: list[Episode] = []
    for episode in episodes:
        if merged and episode.start - merged[-1].end <= _MERGE_GAP:
            merged[-1] = Episode(merged[-1].start, episode.end)
            continue
        merged.append(episode)
    return merged


@dataclass(frozen=True)
class EventScore:
    """イベント単位の成績。"""

    truth_count: int  # 正解の危険エピソード数
    detected: int  # そのうち警告できた数
    latencies: tuple[float, ...]  # 検出できたエピソードごとの遅延（秒）
    false_alarms: int  # どの危険とも重ならなかった警告の数
    safe_seconds: float  # 平常だった時間の合計（誤警告率の分母）

    @property
    def detection_rate(self) -> float:
        return self.detected / self.truth_count if self.truth_count else 0.0

    @property
    def median_latency(self) -> float:
        """検出遅延の中央値。1つも検出できていなければ nan。

        平均ではなく中央値で見る。1回の大外れが平均を支配し、いつもの反応の速さが
        分からなくなるため。
        """
        if not self.latencies:
            return float("nan")
        values = sorted(self.latencies)
        mid = len(values) // 2
        if len(values) % 2:
            return values[mid]
        return (values[mid - 1] + values[mid]) / 2.0

    @property
    def false_alarms_per_hour(self) -> float:
        """平常 1 時間あたりの誤警告回数。実運用で最初に効く数字。"""
        if self.safe_seconds <= 0:
            return 0.0
        return self.false_alarms * 3600.0 / self.safe_seconds


def score_events(
    truth: Sequence[Episode],
    alerts: Sequence[Episode],
    total_seconds: float,
) -> EventScore:
    """正解の危険区間と、判定が出した警告区間を突き合わせる。

    検出＝危険区間と重なる警告が1つ以上あること。遅延は危険の開始から、重なった
    最初の警告の開始まで。警告が危険より先に始まっていた場合の遅延は 0（負にしない。
    負の遅延を混ぜると中央値が「早く気づけている」と読めてしまうが、実際には
    危険が始まる前から鳴っていた＝そのぶん誤警告に近い振る舞い）。
    """
    latencies: list[float] = []
    detected = 0
    matched: set[int] = set()
    for episode in truth:
        hits = [i for i, alert in enumerate(alerts) if alert.overlaps(episode)]
        if not hits:
            continue
        detected += 1
        matched.update(hits)
        latencies.append(max(0.0, alerts[hits[0]].start - episode.start))

    danger_seconds = sum(e.duration for e in truth)
    safe_seconds = max(0.0, total_seconds - danger_seconds)
    return EventScore(
        truth_count=len(truth),
        detected=detected,
        latencies=tuple(latencies),
        false_alarms=len(alerts) - len(matched),
        safe_seconds=safe_seconds,
    )


def format_event_score(score: EventScore) -> str:
    latency = score.median_latency
    latency_text = "—" if latency != latency else f"{latency:.1f}s"  # nan は自分と等しくない
    return "\n".join(
        [
            f"危険エピソード: {score.truth_count} 件 / 検出 {score.detected} 件"
            f"（{score.detection_rate:.0%}）",
            f"検出遅延（中央値）: {latency_text}",
            f"誤警告: {score.false_alarms} 回 / 平常 {score.safe_seconds / 60:.1f} 分"
            f" = {score.false_alarms_per_hour:.1f} 回/時",
        ]
    )
