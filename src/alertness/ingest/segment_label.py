"""フレーム時刻から軸別の正解ラベルを解決するラベル供給器。

csv_sink は毎フレーム、ラベル供給器から `drowsiness` / `distraction` を読んで書き込む。
ここではその2つを、フレームの時刻に対応する区間の段階ラベルへ差し替える。LabelState を
継承しているので、旧・単一 label 列（.value）は空のまま互換を保つ。
"""

from __future__ import annotations

from ..labeling import LabelState
from .manifest import ClipManifest


class SegmentLabelProvider(LabelState):
    def __init__(self, manifest: ClipManifest) -> None:
        super().__init__("")
        self._manifest = manifest
        self.drowsiness = ""
        self.distraction = ""

    def apply(self, timestamp: float) -> None:
        labels = self._manifest.labels_at(timestamp)
        self.drowsiness = labels["drowsiness"]
        self.distraction = labels["distraction"]
