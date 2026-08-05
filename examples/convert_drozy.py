"""DROZY データセット → manifest 変換器。

この実装は、DROZY の PSG・PVT・KSS を既存の manifest 形式へ落とすための入口です。
初期版では、PSG を簡易特徴量に変換し、LoD を生成したうえで、区間圧縮して manifest を書き出します。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from alertness.bio.psg import build_psg_feature_series, read_psg_signal
from alertness.calibration.baseline import calibrate_with_pvt_kss, normalize_feature_series
from alertness.classifier.cds import compute_cds
from alertness.classifier.lod import classify_lod
from alertness.ingest.mapping import segment, write_manifest
from alertness.temporal import build_manifest_segments, smooth_labels


def discover_sessions(root: Path) -> list[dict[str, Any]]:
    """DROZY の実データ構造に合わせて session を発見する。"""
    sessions: list[dict[str, Any]] = []
    if not root.exists():
        return sessions

    psg_dir = root / "psg"
    pvt_dir = root / "pvt-rt"
    timestamps_dir = root / "timestamps"
    videos_dir = root / "videos_i8"

    psg_files: dict[str, Path] = {}
    pvt_files: dict[str, Path] = {}
    timestamp_files: dict[str, Path] = {}
    video_files: dict[str, Path] = {}

    if any([psg_dir.exists(), pvt_dir.exists(), timestamps_dir.exists(), videos_dir.exists()]):
        psg_files = {p.stem: p for p in psg_dir.glob("*.edf") if p.is_file()} if psg_dir.exists() else {}
        pvt_files = {p.stem: p for p in pvt_dir.glob("*.csv") if p.is_file()} if pvt_dir.exists() else {}
        timestamp_files = {p.stem: p for p in timestamps_dir.glob("*.txt") if p.is_file()} if timestamps_dir.exists() else {}
        video_files = {p.stem: p for p in videos_dir.glob("*.mp4") if p.is_file()} if videos_dir.exists() else {}

    if not (psg_files or pvt_files or timestamp_files or video_files):
        for subject_dir in sorted(root.iterdir()):
            if not subject_dir.is_dir():
                continue
            for session_dir in sorted(subject_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                files = {p.name: p for p in session_dir.iterdir() if p.is_file()}
                if not files:
                    continue
                video_path = next((p for p in files.values() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}), None)
                eeg_path = next((p for p in files.values() if "eeg" in p.name.lower()), None)
                eog_path = next((p for p in files.values() if "eog" in p.name.lower()), None)
                pvt_path = next((p for p in files.values() if "pvt" in p.name.lower()), None)
                sessions.append(
                    {
                        "subject": subject_dir.name,
                        "session": session_dir.name,
                        "root": session_dir,
                        "video": video_path,
                        "video_fps": None,
                        "eeg": eeg_path,
                        "eog": eog_path,
                        "psg": eeg_path or eog_path,
                        "pvt": pvt_path,
                        "timestamps": None,
                        "kss": None,
                    }
                )
        return sessions

    session_ids = sorted(set(psg_files) | set(pvt_files) | set(timestamp_files) | set(video_files))
    for session_id in session_ids:
        psg_path = psg_files.get(session_id)
        pvt_path = pvt_files.get(session_id)
        timestamp_path = timestamp_files.get(session_id)
        video_path = video_files.get(session_id)
        if not (psg_path or pvt_path or timestamp_path or video_path):
            continue

        subject_name = session_id.split("-", 1)[0] if "-" in session_id else session_id
        sessions.append(
            {
                "subject": subject_name or session_id,
                "session": session_id,
                "root": root,
                "video": video_path,
                "video_fps": None,
                "eeg": psg_path,
                "eog": psg_path,
                "psg": psg_path,
                "pvt": pvt_path,
                "timestamps": timestamp_path,
                "kss": None,
            }
        )
    return sessions


def _read_signal(path: Path | None) -> list[float]:
    if path is None or not path.exists():
        return []
    signal, _ = read_psg_signal(path)
    return signal.tolist()


def _read_pvt_series(path: Path | None) -> list[float]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    values: list[float] = []
    for row in rows:
        if not row:
            continue
        try:
            values.append(float(row[0]))
        except ValueError:
            continue
    return values


def _read_timestamp_series(path: Path | None) -> list[float]:
    if path is None or not path.exists():
        return []
    values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                values.append(float(parts[-1]))
            except ValueError:
                continue
    return values


def infer_video_fps(session: dict[str, Any]) -> float:
    """セッション情報から動画 FPS を推定する。"""
    explicit = session.get("video_fps")
    if explicit is not None:
        try:
            fps = float(explicit)
        except (TypeError, ValueError):
            fps = 0.0
        if fps > 0:
            return fps

    video = session.get("video")
    if isinstance(video, Path):
        name = video.name.lower()
        if "15" in name and "fps" in name:
            return 15.0
        if "30" in name and "fps" in name:
            return 30.0

    context = str(session.get("session", "")).lower()
    if "15" in context:
        return 15.0
    if "30" in context:
        return 30.0
    return 30.0


def build_manifest_for_session(session: dict[str, Any]) -> dict[str, Any]:
    """セッション単位で LO D を生成し、manifest へ変換する。"""
    eeg = _read_signal(session.get("eeg"))
    eog = _read_signal(session.get("eog"))
    if not eeg or not eog:
        raise ValueError(f"PSG signal not found for {session['subject']}/{session['session']}")

    pvt_values = _read_pvt_series(session.get("pvt"))
    timestamps = _read_timestamp_series(session.get("timestamps"))
    features = build_psg_feature_series(eeg, eog, sample_rate=512, window_seconds=1.0)
    normalized = normalize_feature_series(features)
    scores = compute_cds(normalized)
    calibrated_scores = calibrate_with_pvt_kss(
        scores,
        pvt=pvt_values,
        kss=timestamps,
    )
    labels = classify_lod(calibrated_scores)
    smoothed = smooth_labels(labels, window=3)

    video_fps = infer_video_fps(session)

    mapped_segments = build_manifest_segments(
        smoothed,
        fps=video_fps,
        min_duration_seconds=1.0,
    )
    manifest_segments = [
        segment(start=float(item["start"]), end=float(item["end"]), drowsiness=str(item["label"]))
        for item in mapped_segments
    ]
    if not manifest_segments:
        manifest_segments = [segment(start=0.0, end=1.0, drowsiness="none")]

    return {
        "video": str(session["video"].name if session.get("video") else "unknown.mp4"),
        "subject": f"drozy_{session['subject']}",
        "context": session["session"],
        "segments": manifest_segments,
    }


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "data/DROZY")
    if not root.exists():
        print(f"データセットのフォルダが見つかりません: {root}", file=sys.stderr)
        return 1

    sessions = discover_sessions(root)
    if not sessions:
        print(f"{root} からセッションを見つけられませんでした。", file=sys.stderr)
        return 1

    out_dir = Path("data/manifests")
    for session in sessions:
        try:
            manifest = build_manifest_for_session(session)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            continue

        manifest_path = out_dir / f"drozy_{session['subject']}_{session['session']}.json"
        write_manifest(
            manifest_path,
            video=manifest["video"],
            subject=manifest["subject"],
            context=manifest["context"],
            segments=manifest["segments"],
        )
        print(f"{session['subject']}/{session['session']}: wrote {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
