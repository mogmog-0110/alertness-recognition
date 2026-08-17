"""外部データセット（動画＋区間ラベル）を、既存の録画CSVと同じ形式に変換する取り込み層。

核が知るのは manifest（動画＋区間ラベルの正規形）だけ。どのデータセットも、
まず manifest の形に直してから渡す。特徴抽出・正規化・CSV出力は本体
（pipeline / csv_sink）をそのまま使うので、収録済みデータと同じ土俵で扱える。
"""

from .manifest import ClipManifest, Segment, load_manifest, manifests_from
from .runner import IngestBatchResult, IngestSkip, run_ingest, run_ingest_all, run_ingest_batch

__all__ = [
    "ClipManifest",
    "IngestBatchResult",
    "IngestSkip",
    "Segment",
    "load_manifest",
    "manifests_from",
    "run_ingest",
    "run_ingest_all",
    "run_ingest_batch",
]
