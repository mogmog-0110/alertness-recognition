"""ガイド付き収録の進行ロジック。

アプリが「いまこの状態にしてください」と具体的に指示し、その間のフレームに
自動でラベルを付ける。指示→保持→次の指示…を指定周回ぶん繰り返す。
保持(hold)中だけラベルを付け、移行(ready)中はラベル無し（採点対象外）にする。
時刻を引数で受ける純粋なロジックにしてあるのでテストできる。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence


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
            "・画面をまっすぐ見る\n"
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
                return GuidedStep(title, prompt.instruction, label, phase, end - elapsed,
                                  elapsed / self._total)
        return GuidedStep("完了", "おつかれさまでした", "", "done", 0.0, 1.0)
