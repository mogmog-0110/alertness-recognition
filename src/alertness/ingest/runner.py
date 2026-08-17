"""動画＋manifest を、既存の録画CSVと同じ形式に変換する取り込み実行部。

2パス構成。1パス目で自動キャリブ用に生特徴量を集め、2パス目で本体の pipeline を
通して正規化特徴量・判定・区間ラベルを1フレーム1行で書き出す。特徴抽出と正規化と
CSV出力は本体の実装をそのまま使うので、収録済みデータと同じ土俵で評価できる。
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

from ..contracts import CalibrationProfile
from .autocalibrate import estimate_profile
from .manifest import ClipManifest
from .segment_label import SegmentLabelProvider


@dataclass(frozen=True)
class IngestSkip:
    """FPS事前検査によって取り込み対象外になった動画と理由。"""

    video: str
    reason: str


@dataclass(frozen=True)
class IngestBatchResult:
    """1回のmanifestバッチで採用したFPS、生成先、スキップ情報。"""

    csv_fps: float | None
    directories: tuple[Path, ...]
    skipped: tuple[IngestSkip, ...]


def _configured_csv_fps(config: dict) -> float | None:
    """``None``をautoとして、ingest.csv_fpsを検証して返す。"""
    ingest = config.get("ingest", {})
    if not isinstance(ingest, dict):
        raise ValueError("ingest設定はマッピングである必要があります")
    value = ingest.get("csv_fps", "auto")
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("ingest.csv_fps は正数または 'auto' を指定してください")
    fps = float(value)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("ingest.csv_fps は有限の正数または 'auto' を指定してください")
    return fps


def _video_source(video: str, csv_fps: float):
    from ..sources.frame_rate import DownsampledFrameSource
    from ..sources.video_file import VideoFileSource

    source = VideoFileSource(video)
    try:
        return DownsampledFrameSource(source, csv_fps)
    except Exception:
        source.close()
        raise


def _collect_raw(config: dict, video: str, csv_fps: float) -> list[dict]:
    from ..factory import build_detector
    from ..features.extractor import FaceFeatureExtractor

    detector = build_detector(config)
    extractor = FaceFeatureExtractor()
    source = _video_source(video, csv_fps)
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
    config: dict,
    manifest: ClipManifest,
    out_base: str | Path = "runs/ingested",
    *,
    csv_fps: float | None = None,
) -> Path:
    """1本の動画を取り込み、書き出したディレクトリを返す。

    ``csv_fps``未指定かつ設定がautoの場合、この動画自身のFPSを採用する。
    """
    from ..factory import build_pipeline, cue_names, dimension_names
    from ..feedback.csv_sink import CsvRecorderSink
    from ..sources.frame_rate import validate_downsample_fps
    from ..sources.video_file import probe_video_fps

    source_fps = probe_video_fps(manifest.video)
    effective_fps = csv_fps
    if effective_fps is None:
        effective_fps = _configured_csv_fps(config) or source_fps
    validate_downsample_fps(source_fps, effective_fps)

    profile = _calibrate(config, manifest, effective_fps)

    pipeline = build_pipeline(config, fps=effective_fps)
    pipeline.set_profile(profile)
    source = _video_source(manifest.video, effective_fps)
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
    """互換API。バッチ検査後の生成先だけを返し、スキップは警告する。"""
    result = run_ingest_batch(config, manifests, out_base)
    for skipped in result.skipped:
        warnings.warn(f"SKIP {skipped.video}: {skipped.reason}", stacklevel=2)
    return list(result.directories)


def run_ingest_batch(
    config: dict, manifests: Iterable[ClipManifest], out_base: str | Path = "runs/ingested"
) -> IngestBatchResult:
    """manifest集合のFPSを事前検査し、共通のCSV FPSで有効動画を取り込む。"""
    from ..sources.frame_rate import validate_downsample_fps
    from ..sources.video_file import probe_video_fps

    requested_fps = _configured_csv_fps(config)
    candidates: list[tuple[ClipManifest, float]] = []
    skipped: list[IngestSkip] = []
    for manifest in manifests:
        try:
            source_fps = probe_video_fps(manifest.video)
            if requested_fps is not None:
                validate_downsample_fps(source_fps, requested_fps)
        except (OSError, RuntimeError, ValueError) as exc:
            skipped.append(IngestSkip(manifest.video, str(exc)))
            continue
        candidates.append((manifest, source_fps))

    if not candidates:
        return IngestBatchResult(None, (), tuple(skipped))

    effective_fps = requested_fps or min(source_fps for _manifest, source_fps in candidates)
    directories = tuple(
        run_ingest(config, manifest, out_base, csv_fps=effective_fps)
        for manifest, _source_fps in candidates
    )
    return IngestBatchResult(effective_fps, directories, tuple(skipped))


def _calibrate(
    config: dict, manifest: ClipManifest, csv_fps: float
) -> CalibrationProfile:
    rows = _collect_raw(config, manifest.video, csv_fps)
    return estimate_profile(rows, manifest.subject)
