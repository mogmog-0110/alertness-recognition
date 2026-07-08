"""動画＋manifest を、既存の録画CSVと同じ形式に変換する取り込み実行部。

2パス構成。1パス目で自動キャリブ用に生特徴量を集め、2パス目で本体の pipeline を
通して正規化特徴量・判定・区間ラベルを1フレーム1行で書き出す。特徴抽出と正規化と
CSV出力は本体の実装をそのまま使うので、収録済みデータと同じ土俵で評価できる。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..contracts import CalibrationProfile
from .autocalibrate import estimate_profile
from .manifest import ClipManifest
from .segment_label import SegmentLabelProvider


def _collect_raw(config: dict, video: str) -> list[dict]:
    from ..factory import build_detector
    from ..features.extractor import FaceFeatureExtractor
    from ..sources.video_file import VideoFileSource

    detector = build_detector(config)
    extractor = FaceFeatureExtractor()
    source = VideoFileSource(video)
    rows: list[dict] = []
    try:
        for frame in source.frames():
            landmarks = detector.detect(frame)
            raw = extractor.extract(landmarks, frame.timestamp)
            if raw.face_present:
                rows.append(dict(raw.values))
    finally:
        source.close()
        detector.close()
    return rows


def _write_rows(pipeline, source, provider: SegmentLabelProvider, sink) -> int:
    written = 0
    for frame in source.frames():
        obs = pipeline.observe(frame)
        assessment = pipeline.classify(obs)
        provider.apply(frame.timestamp)
        sink.emit(obs, assessment)
        written += 1
    return written


def _out_dir(base: str | Path, manifest: ClipManifest) -> Path:
    # クリップごとに別ディレクトリへ。csv_sink の時刻ベース命名の衝突を避ける。
    stem = Path(manifest.video).stem
    directory = Path(base) / f"{manifest.subject}__{stem}"
    return directory


def run_ingest(
    config: dict, manifest: ClipManifest, out_base: str | Path = "runs/ingested"
) -> Path:
    """1本の動画を取り込み、書き出したディレクトリを返す。"""
    from ..factory import build_pipeline, cue_names, dimension_names
    from ..feedback.csv_sink import CsvRecorderSink
    from ..sources.video_file import VideoFileSource

    profile = _calibrate(config, manifest)

    pipeline = build_pipeline(config)
    pipeline.set_profile(profile)
    source = VideoFileSource(manifest.video)
    provider = SegmentLabelProvider(manifest)
    directory = _out_dir(out_base, manifest)
    sink = CsvRecorderSink(
        str(directory),
        dimension_names(config),
        provider,
        manifest.subject,
        cue_names(config),
        manifest.context,
    )
    try:
        _write_rows(pipeline, source, provider, sink)
    finally:
        sink.close()
        source.close()
        pipeline.close()
    return directory


def run_ingest_all(
    config: dict, manifests: Iterable[ClipManifest], out_base: str | Path = "runs/ingested"
) -> list[Path]:
    """manifest の列を順に取り込む。列の作り手（ファイル/ダミー生成など）は問わない。"""
    return [run_ingest(config, m, out_base) for m in manifests]


def _calibrate(config: dict, manifest: ClipManifest) -> CalibrationProfile:
    rows = _collect_raw(config, manifest.video)
    return estimate_profile(rows, manifest.subject)
