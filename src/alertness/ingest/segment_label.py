"""フレーム時刻から軸別の正解ラベルを解決するラベル供給器。

csv_sink は毎フレーム、ラベル供給器の `levels`（軸名→段階）を読んで軸別ラベル列に書き込む。
ここでは時刻に対応する区間の段階ラベルへ差し替える。付いていない軸は levels に現れず、
csv 側で空（未アノテ）になる。LabelState を継承しているので、旧・単一 label 列（.value）は
空のまま互換を保つ。
"""

from __future__ import annotations

from ..labeling import LabelState
from .manifest import ClipManifest


class SegmentLabelProvider(LabelState):
    def __init__(self, manifest: ClipManifest) -> None:
        super().__init__("")
        self._manifest = manifest
        self.levels: dict[str, str] = {}

    def apply(self, timestamp: float) -> None:
        self.levels = self._manifest.labels_at(timestamp)
