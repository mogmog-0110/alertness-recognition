"""DROZY の PSG を解析し、眠気区間を既存 ingest 用 manifest として出力する。

この変換器は、公式データツリーにある動画、EDF（EEG/EOG）、PVT、動画 timestamps、KSS を
セッションIDで対応付ける。EDFから得た窓単位の特徴量を被験者の PVT1 冒頭で基準化し、CDS
（連続眠気スコア）へ統合した後、PVT1からの反応悪化量で LoD の段階境界を補正する。短い段階
変化を時間方向に平滑化し、最終的に drowsiness 軸だけを持つ区間 manifest を生成する。

PSG が眠気ラベルの主な根拠で、PVT は被験者・セッション差を反映する境界校正にだけ使う。
主観指標の KSS は生成ラベルを直接変更せず、出力時の方向性検証に使う。PVT1、必須PSG
チャンネル、動画同期情報のいずれかが欠けるセッションは補完せずスキップする。この方針と
段階の意味は docs/annotation-guide.md、窓幅・特徴重み・境界・平滑化値は
config/default.yaml の ``drozy`` セクションを正準とする。

生成物は ``python -m alertness.ingest`` に渡し、動画特徴量と区間ラベルを持つ Canonical CSV
へ変換する。通常のWebカメラ推論経路からは呼ばれず、DROZYを教師データ化するときだけ使う。
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alertness.bio.psg import PsgFeature, extract_psg_features, read_psg
from alertness.bio.pvt import (
    PvtSummary,
    impairment_from_baseline,
    read_pvt,
    summarize_pvt,
    summarize_pvt_windows,
)
from alertness.calibration.baseline import BaselineStats, fit_baseline, normalize_features
from alertness.classifier.cds import DEFAULT_WEIGHTS, compute_cds
from alertness.classifier.lod import calibrate_thresholds
from alertness.config import load_config
from alertness.ingest.mapping import segment, write_manifest
from alertness.temporal import smooth_lod_segments

# drowsiness は用途非依存の軸（colab の CONTEXT_FREE_AXES）なので、空でも学習に影響しない。
CONTEXT = ""


@dataclass(frozen=True)
class DrozySession:
    subject: str
    test: int
    session_id: str
    video: Path | None
    psg: Path | None
    pvt: Path | None
    timestamps: Path | None
    kss: float | None

    def __getitem__(self, key: str) -> Any:
        """以前のdict型セッションを参照する利用者向けの読み取り互換。"""
        aliases = {"session": "session_id", "eeg": "psg", "eog": "psg"}
        return getattr(self, aliases.get(key, key))


@dataclass(frozen=True)
class SessionAnalysis:
    session: DrozySession
    features: tuple[PsgFeature, ...]
    duration_seconds: float
    pvt_summary: PvtSummary
    pvt_windows: tuple[tuple[float, PvtSummary], ...] = ()


@dataclass(frozen=True)
class ConversionResult:
    manifest: dict[str, Any]
    mean_cds: float
    pvt_impairment: float
    kss: float | None


def _files_by_stem(directory: Path, pattern: str) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {
        path.stem: path
        for path in directory.glob(pattern)
        if path.is_file() and not path.name.startswith((".", "._"))
    }


def read_kss(root: Path) -> dict[tuple[str, int], float | None]:
    """KSS.txtの14行×3列を被験者・テストへ対応付ける。0は欠損。"""
    path = root / "KSS.txt"
    if not path.is_file():
        return {}
    output: dict[tuple[str, int], float | None] = {}
    subject = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        values: list[float] = []
        for token in line.replace(",", " ").split():
            try:
                values.append(float(token))
            except ValueError:
                continue
        if not values:
            continue
        subject += 1
        for test, value in enumerate(values[:3], start=1):
            output[(str(subject), test)] = value if 1.0 <= value <= 9.0 else None
    return output


def _parse_session_id(session_id: str) -> tuple[str, int] | None:
    parts = session_id.rsplit("-", 1)
    if len(parts) != 2:
        return None
    try:
        test = int(parts[1])
    except ValueError:
        return None
    if test not in (1, 2, 3):
        return None
    return parts[0], test


def discover_sessions(root: Path) -> list[DrozySession]:
    """公式のフラットなDROZYツリーから、存在する全セッション候補を発見する。"""
    psg = _files_by_stem(root / "psg", "*.edf")
    pvt = _files_by_stem(root / "pvt-rt", "*.csv")
    timestamps = _files_by_stem(root / "timestamps", "*.txt")
    videos = _files_by_stem(root / "videos_i8", "*.mp4")
    kss = read_kss(root)
    session_ids = sorted(set(psg) | set(pvt) | set(timestamps) | set(videos))
    sessions: list[DrozySession] = []
    for session_id in session_ids:
        parsed = _parse_session_id(session_id)
        if parsed is None:
            continue
        subject, test = parsed
        sessions.append(
            DrozySession(
                subject=subject,
                test=test,
                session_id=session_id,
                video=videos.get(session_id),
                psg=psg.get(session_id),
                pvt=pvt.get(session_id),
                timestamps=timestamps.get(session_id),
                kss=kss.get((subject, test)),
            )
        )
    return sessions


def read_video_timestamps(path: Path) -> list[float]:
    """各行の末尾にある開始時からの経過ミリ秒を秒へ変換する。"""
    values: list[float] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().replace(",", " ").split()
        if not parts:
            continue
        try:
            elapsed_ms = float(parts[-1])
        except ValueError:
            continue
        values.append(elapsed_ms / 1000.0)
    if len(values) < 2:
        raise ValueError(f"動画timestampsが2件以上ありません: {path}")
    origin = values[0]
    normalized = [value - origin for value in values]
    if any(
        current <= previous
        for previous, current in zip(normalized[:-1], normalized[1:], strict=True)
    ):
        raise ValueError(f"動画timestampsが単調増加ではありません: {path}")
    return normalized


def _video_duration(path: Path) -> float | None:
    try:
        import cv2
    except ImportError:  # pragma: no cover - base dependency
        return None
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or frames <= 0:
        return None
    return frames / fps


def _required_paths(session: DrozySession) -> tuple[Path, Path, Path, Path]:
    missing = [
        name for name in ("video", "psg", "pvt", "timestamps") if getattr(session, name) is None
    ]
    if missing:
        raise ValueError(f"必須モダリティがありません: {', '.join(missing)}")
    return session.video, session.psg, session.pvt, session.timestamps  # type: ignore[return-value]


def _feature_options(config: Mapping[str, Any]) -> dict[str, Any]:
    drozy = config.get("drozy", {})
    eeg = drozy.get("eeg", {})
    eog = drozy.get("eog", {})
    return {
        "window_seconds": float(drozy.get("window_seconds", 10.0)),
        "stride_seconds": float(drozy.get("stride_seconds", 1.0)),
        "eeg_low_hz": float(eeg.get("low_hz", 0.5)),
        "eeg_high_hz": float(eeg.get("high_hz", 35.0)),
        "eog_low_hz": float(eog.get("low_hz", 0.1)),
        "eog_high_hz": float(eog.get("high_hz", 15.0)),
        "eog_event_z": float(eog.get("event_z", 2.0)),
        "blink_min_seconds": float(eog.get("blink_min_seconds", 0.08)),
        "blink_max_seconds": float(eog.get("blink_max_seconds", 0.8)),
        "microsleep_min_seconds": float(eog.get("microsleep_min_seconds", 0.5)),
    }


def analyze_session(session: DrozySession, config: Mapping[str, Any]) -> SessionAnalysis:
    video, psg_path, pvt_path, timestamp_path = _required_paths(session)
    drozy = config.get("drozy", {})
    recording = read_psg(psg_path, channel_aliases=drozy.get("channel_aliases"))
    timestamps = read_video_timestamps(timestamp_path)
    durations = [recording.duration_seconds, timestamps[-1]]
    video_duration = _video_duration(video)
    if video_duration is not None:
        durations.append(video_duration)
    duration = min(durations)
    features = tuple(
        feature
        for feature in extract_psg_features(recording, **_feature_options(config))
        if feature.timestamp <= duration and feature.valid
    )
    if len(features) < 2:
        raise ValueError("同期後に有効なPSG特徴量が2件以上ありません")
    pvt_config = drozy.get("pvt", {})
    pvt_samples = read_pvt(pvt_path)
    false_start_ms = float(pvt_config.get("false_start_ms", 100.0))
    lapse_ms = float(pvt_config.get("lapse_ms", 500.0))
    pvt_summary = summarize_pvt(
        pvt_samples,
        false_start_ms=false_start_ms,
        lapse_ms=lapse_ms,
    )
    pvt_windows = tuple(
        summarize_pvt_windows(
            pvt_samples,
            window_seconds=float(pvt_config.get("window_seconds", 20.0)),
            false_start_ms=false_start_ms,
            lapse_ms=lapse_ms,
        )
    )
    return SessionAnalysis(
        session,
        features,
        duration,
        pvt_summary,
        pvt_windows,
    )


def build_manifest_for_session(
    analysis: SessionAnalysis,
    baseline: BaselineStats,
    pvt_baseline: PvtSummary,
    config: Mapping[str, Any],
) -> ConversionResult:
    drozy = config.get("drozy", {})
    cds_config = drozy.get("cds", {})
    normalized = normalize_features(analysis.features, baseline)
    scores = compute_cds(
        normalized,
        weights={
            name: float(value) for name, value in cds_config.get("weights", DEFAULT_WEIGHTS).items()
        },
        sigmoid_center=float(cds_config.get("sigmoid_center", 0.0)),
        sigmoid_scale=float(cds_config.get("sigmoid_scale", 1.0)),
    )
    impairment = impairment_from_baseline(analysis.pvt_summary, pvt_baseline)
    lod = drozy.get("lod", {})
    thresholds = calibrate_thresholds(
        [float(value) for value in lod.get("thresholds", (20.0, 50.0, 75.0))],
        impairment,
        gain=float(lod.get("pvt_gain", 5.0)),
        max_shift=float(lod.get("pvt_max_shift", 10.0)),
    )
    temporal = drozy.get("temporal", {})
    mapped = smooth_lod_segments(
        scores,
        thresholds=thresholds,
        timestamps=[feature.timestamp for feature in analysis.features],
        duration_seconds=analysis.duration_seconds,
        stride_seconds=float(drozy.get("stride_seconds", 1.0)),
        median_seconds=float(temporal.get("median_seconds", 5.0)),
        hysteresis_margin=float(temporal.get("hysteresis_margin", 5.0)),
        min_duration_seconds=float(temporal.get("min_duration_seconds", 5.0)),
    )
    if not mapped:
        raise ValueError("有効なLoD区間を生成できませんでした")
    manifest_segments = [
        segment(
            float(item["start"]),
            float(item["end"]),
            drowsiness=str(item["label"]),
        )
        for item in mapped
    ]
    manifest = {
        "video": analysis.session.video.as_posix(),
        "subject": f"drozy_{analysis.session.subject}",
        "context": CONTEXT,
        "segments": manifest_segments,
    }
    return ConversionResult(
        manifest=manifest,
        mean_cds=sum(scores) / len(scores),
        pvt_impairment=impairment,
        kss=analysis.session.kss,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DROZY PSGから眠気manifestを生成する")
    parser.add_argument("root", type=Path, help="DROZYデータセットのルート")
    parser.add_argument("--config", default="config/default.yaml", help="設定ファイル")
    parser.add_argument("--out", type=Path, default=Path("data/manifests"), help="出力先")
    parser.add_argument("--subject", help="指定した被験者だけ変換")
    parser.add_argument("--force", action="store_true", help="既存manifestを上書き")
    return parser


def _pearson(rows: Sequence[tuple[float, float]]) -> float | None:
    if len(rows) < 2:
        return None
    left_mean = sum(left for left, _right in rows) / len(rows)
    right_mean = sum(right for _left, right in rows) / len(rows)
    numerator = sum((left - left_mean) * (right - right_mean) for left, right in rows)
    left_scale = math.sqrt(sum((left - left_mean) ** 2 for left, _right in rows))
    right_scale = math.sqrt(sum((right - right_mean) ** 2 for _left, right in rows))
    if left_scale <= 1e-12 or right_scale <= 1e-12:
        return None
    return numerator / (left_scale * right_scale)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.root.is_dir():
        print(f"データセットのフォルダが見つかりません: {args.root}")
        return 1
    config = load_config(args.config)
    sessions = discover_sessions(args.root)
    if args.subject:
        sessions = [session for session in sessions if session.subject == str(args.subject)]
    grouped: dict[str, list[DrozySession]] = defaultdict(list)
    for session in sessions:
        grouped[session.subject].append(session)

    written = 0
    skipped = 0
    validation: list[tuple[float, float | None, float]] = []
    for subject, subject_sessions in sorted(grouped.items()):
        analyzed: dict[int, SessionAnalysis] = {}
        for session in sorted(subject_sessions, key=lambda item: item.test):
            try:
                analyzed[session.test] = analyze_session(session, config)
            except (ImportError, OSError, ValueError) as exc:
                print(f"SKIP {session.session_id}: {exc}")
                skipped += 1
        if 1 not in analyzed:
            print(f"SKIP subject {subject}: PVT1基準を作成できません")
            skipped += len(analyzed)
            continue
        if analyzed[1].pvt_summary.mean_reaction_ms is None:
            print(f"SKIP subject {subject}: PVT1に通常反応がなく、PVT基準を作成できません")
            skipped += len(analyzed)
            continue
        try:
            baseline = fit_baseline(
                analyzed[1].features,
                baseline_seconds=float(config.get("drozy", {}).get("baseline_seconds", 120.0)),
            )
        except ValueError as exc:
            print(f"SKIP subject {subject}: {exc}")
            skipped += len(analyzed)
            continue
        for _test, analysis in sorted(analyzed.items()):
            out_path = args.out / f"drozy_{subject}_{analysis.session.session_id}.json"
            if out_path.exists() and not args.force:
                print(f"SKIP {analysis.session.session_id}: 既存 {out_path}")
                skipped += 1
                continue
            try:
                result = build_manifest_for_session(
                    analysis, baseline, analyzed[1].pvt_summary, config
                )
                write_manifest(out_path, **result.manifest)
            except (OSError, ValueError) as exc:
                print(f"SKIP {analysis.session.session_id}: {exc}")
                skipped += 1
                continue
            written += 1
            validation.append((result.mean_cds, result.kss, result.pvt_impairment))
            pvt = analysis.pvt_summary
            mean_rt = (
                "欠損" if pvt.mean_reaction_ms is None else f"{pvt.mean_reaction_ms:.1f}ms"
            )
            print(
                f"WROTE {out_path}: CDS={result.mean_cds:.1f}, "
                f"PVT悪化={result.pvt_impairment:+.3f}, KSS={result.kss}, "
                f"PVT=[総数={pvt.valid_count + pvt.false_start_count}, "
                f"通常={pvt.normal_count}, false start={pvt.false_start_count}, "
                f"lapse={pvt.lapse_count}, 平均RT={mean_rt}, lapse率={pvt.lapse_rate:.3f}]"
            )
    if validation:
        kss_rows = [row for row in validation if row[1] is not None]
        print(f"検証対象: PVT={len(validation)}セッション, KSS={len(kss_rows)}セッション")
        pvt_correlation = _pearson([(cds, impairment) for cds, _kss, impairment in validation])
        kss_correlation = _pearson([(cds, float(kss)) for cds, kss, _pvt in kss_rows])
        pvt_text = "算出不可" if pvt_correlation is None else f"{pvt_correlation:+.3f}"
        kss_text = "算出不可" if kss_correlation is None else f"{kss_correlation:+.3f}"
        print(f"方向性相関: CDS-PVT={pvt_text}, CDS-KSS={kss_text}（正が期待方向）")
    print(f"完了: 生成={written}, スキップ={skipped}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
