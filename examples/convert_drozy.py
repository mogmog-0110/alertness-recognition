"""DROZY データセット → manifest 変換器。

この実装は、DROZY の PSG・PVT・KSS を既存の manifest 形式へ落とすための入口です。
初期版では、PSG を簡易特徴量に変換し、LoD を生成したうえで、区間圧縮して manifest を書き出します。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from alertness.bio.psg import build_psg_feature_series
from alertness.calibration.baseline import normalize_feature_series
from alertness.classifier.cds import compute_cds
from alertness.classifier.lod import classify_lod
from alertness.ingest.mapping import segment, write_manifest
from alertness.temporal import compress_segments, smooth_labels


def discover_sessions(root: Path) -> list[dict[str, Any]]:
    """DROZY の典型構造を仮定して session を発見する。"""
    sessions: list[dict[str, Any]] = []
    if not root.exists():
        return sessions

    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            files = {p.name: p for p in session_dir.iterdir() if p.is_file()}
            if not files:
                continue
            sessions.append(
                {
                    "subject": subject_dir.name,
                    "session": session_dir.name,
                    "root": session_dir,
                    "video": next((p for p in files.values() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}), None),
                    "eeg": next((p for p in files.values() if "eeg" in p.name.lower()), None),
                    "eog": next((p for p in files.values() if "eog" in p.name.lower()), None),
                    "pvt": next((p for p in files.values() if "pvt" in p.name.lower()), None),
                    "kss": next((p for p in files.values() if "kss" in p.name.lower()), None),
                }
            )
    return sessions


def _read_signal(path: Path | None) -> list[float]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        return []
    return [float(line) for line in lines if line.replace("-", "", 1).replace(".", "", 1).isdigit()]


def build_manifest_for_session(session: dict[str, Any]) -> dict[str, Any]:
    """セッション単位で LO D を生成し、manifest へ変換する。"""
    eeg = _read_signal(session.get("eeg"))
    eog = _read_signal(session.get("eog"))
    if not eeg or not eog:
        raise ValueError(f"PSG signal not found for {session['subject']}/{session['session']}")

    features = build_psg_feature_series(eeg, eog, sample_rate=512, window_seconds=1.0)
    normalized = normalize_feature_series(features)
    scores = compute_cds(normalized)
    labels = classify_lod(scores)
    smoothed = smooth_labels(labels, window=3)
    segments = compress_segments(smoothed, min_duration=2)

    manifest_segments = [
        segment(start=float(item["start"]), end=float(item["end"]), drowsiness=str(item["label"]))
        for item in segments
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
