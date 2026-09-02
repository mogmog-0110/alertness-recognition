"""ガイド付き収録の進行ロジック。

アプリが「いまこの状態にしてください」と具体的に指示し、その間のフレームに
自動でラベルを付ける。指示→保持→次の指示…を指定周回ぶん繰り返す。
保持(hold)中だけラベルを付け、移行(ready)中はラベル無し（採点対象外）にする。
時刻を引数で受ける純粋なロジックにしてあるのでテストできる。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    label: str  # この間に記録する正解ラベル
    title: str  # 画面見出し
    instruction: str  # 具体的な指示（複数行可）
    hold_seconds: float = 12.0  # 保持してもらう時間
    ready_seconds: float = 3.0  # 次の状態へ移る準備時間


@dataclass(frozen=True)
class GuidedStep:
    title: str
    instruction: str
    label: str  # ready 中は ""（記録はするが採点対象外）
    phase: str  # "ready" / "hold" / "done"
    remaining: float
    progress: float  # 全体進捗 0..1


# 既定の指示。目・口・頭の状態まで具体的に書く。
DEFAULT_PROMPTS = (
    Prompt(
        label="awake",
        title="覚醒（ふつうの状態）",
        instruction=(
            "・普段の運転で見る方向（前方）を見る\n"
            "・目はしっかり開ける\n"
            "・自然なまばたき\n"
            "・口は閉じる／頭はまっすぐ"
        ),
    ),
    Prompt(
        label="drowsiness",
        title="眠い状態",
        instruction=(
            "・まぶたを半分くらいまで下げる\n"
            "・ゆっくり長めにまばたき\n"
            "・ときどき大きくあくび\n"
            "・頭を少し前に下げる"
        ),
    ),
    Prompt(
        label="distraction",
        title="注意散漫（よそ見）",
        instruction=(
            "・視線を画面の外（左右）へ\n"
            "・顔も左右に向ける\n"
            "・スマホを見るように下や横を向く\n"
            "・画面を見続けない"
        ),
    ),
)


# ストレス収録用の指示。眠気・注意逸脱は「そう見える顔」を演じてもらえば足りるが、
# ストレスは演技では心拍が動かないので、実際に負荷をかけて誘発するしかない。
# 社会的評価 + 時間圧 + 暗算という組み合わせは TSST（Trier Social Stress Test）の骨格で、
# UBFC-Phys など公開データセットのストレス区間もこの方式で作られている。
# ここで付くラベルは「その区間でストレスをかけた」という条件であって、本人の内的状態の
# 測定値ではない。学習・評価はその前提で扱うこと。
STRESS_PROMPTS = (
    Prompt(
        label="awake",
        title="安静（基準）",
        instruction=(
            "・楽な姿勢で、普段の運転で見る方向を見る\n"
            "・普通に呼吸する。話さない\n"
            "・頭を動かさない（rPPG が壊れます）"
        ),
        hold_seconds=120.0,
        ready_seconds=5.0,
    ),
    Prompt(
        label="stress",
        title="暗算（時間を計っています）",
        instruction=(
            "・1022 から 13 を引き続け、声に出す\n"
            "・できるだけ速く。間違えたら 1022 から やり直し\n"
            "・記録者は正誤を見ています\n"
            "・頭は動かさない"
        ),
        hold_seconds=120.0,
        ready_seconds=5.0,
    ),
    Prompt(
        label="awake",
        title="回復（安静に戻す）",
        instruction="・楽にして画面を見る\n・普通に呼吸する。話さない\n・頭を動かさない",
        hold_seconds=120.0,
        ready_seconds=5.0,
    ),
)

# --protocol で選ぶ指示セット。
PROTOCOLS = {
    "acted": DEFAULT_PROMPTS,  # 演技でよい軸（眠気・注意逸脱）
    "stress": STRESS_PROMPTS,  # 実際に負荷をかけて誘発する軸（ストレス）
}


class GuidedSession:
    def __init__(self, prompts: Sequence[Prompt], rounds: int = 3) -> None:
        self._segments: list[tuple[float, float, str, Prompt]] = []
        cursor = 0.0
        for _ in range(max(1, rounds)):
            for prompt in prompts:
                self._segments.append((cursor, cursor + prompt.ready_seconds, "ready", prompt))
                cursor += prompt.ready_seconds
                self._segments.append((cursor, cursor + prompt.hold_seconds, "hold", prompt))
                cursor += prompt.hold_seconds
        self._total = cursor
        self._start: float | None = None

    def step(self, now: float) -> GuidedStep:
        if self._start is None:
            self._start = now
        elapsed = now - self._start
        if elapsed >= self._total:
            return GuidedStep("完了", "おつかれさまでした", "", "done", 0.0, 1.0)

        for start, end, phase, prompt in self._segments:
            if start <= elapsed < end:
                label = prompt.label if phase == "hold" else ""
                title = prompt.title if phase == "hold" else f"次: {prompt.title}"
                return GuidedStep(
                    title, prompt.instruction, label, phase, end - elapsed, elapsed / self._total
                )
        return GuidedStep("完了", "おつかれさまでした", "", "done", 0.0, 1.0)
