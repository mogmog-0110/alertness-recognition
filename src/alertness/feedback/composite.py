"""複数の出力先を1つにまとめる。表示と録画を同時に行うときなどに使う。"""

from __future__ import annotations

from collections.abc import Sequence

from ..contracts import Assessment, Observation
from ..ports import FeedbackSink


class CompositeSink:
    def __init__(self, sinks: Sequence[FeedbackSink]) -> None:
        self._sinks = list(sinks)

    def emit(self, obs: Observation, assessment: Assessment) -> None:
        for sink in self._sinks:
            sink.emit(obs, assessment)

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()
