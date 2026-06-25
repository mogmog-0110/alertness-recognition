"""録画した特徴量を、現在のルール設定で「再判定」する。

録り直さずにしきい値や cue の変更を試せる。CSV に残った特徴量から各フレームの
Observation を組み直し、設定から作った分類器に通す。セッションごとに時系列の
連続性（PERCLOS など）と EMA を保つため、セッション単位で分類器を作り直す。
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .. import factory
from ..contracts import CalibrationProfile, FaceLandmarks, Features, Frame, Observation
from ..temporal import TemporalContext

_META = {"session_id", "subject", "label", "face_present", "timestamp"}
_DUMMY = np.zeros((2, 2, 3), dtype=np.uint8)


def _row_to_features(row: dict[str, str]) -> Features:
    values: dict[str, float] = {}
    for key, raw in row.items():
        if key in _META or key.startswith("dim_") or key.startswith("cue_") or raw == "":
            continue
        try:
            values[key] = float(raw)
        except ValueError:
            continue
    face_present = (row.get("face_present") or "1") not in ("0", "")
    return Features(values=values, timestamp=float(row.get("timestamp") or 0.0),
                    face_present=face_present)


def _observation(features: Features, temporal: TemporalContext) -> Observation:
    frame = Frame(image=_DUMMY, index=0, timestamp=features.timestamp)
    landmarks = FaceLandmarks(np.zeros((0, 3)), (2, 2), features.face_present)
    return Observation(frame, landmarks, features, temporal, CalibrationProfile.identity())


def _state(assessment, min_level: int, awake: str) -> str:
    best: str | None = None
    best_level = -1
    for name, dim in assessment.dimensions.items():
        if int(dim.level) > best_level:
            best_level = int(dim.level)
            best = name
    return best if best is not None and best_level >= min_level else awake


def replay_predict(
    paths: Sequence[str], config: dict[str, Any], awake: str = "awake", min_level: int = 2
) -> tuple[list[str], list[str]]:
    fps = config.get("camera", {}).get("target_fps", 30)
    y_true: list[str] = []
    y_pred: list[str] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in csv.DictReader(f):
                sessions[row.get("session_id", "")].append(row)

        for rows in sessions.values():
            rows.sort(key=lambda r: float(r.get("timestamp") or 0.0))
            classifier = factory.build_classifier(config)  # セッションごとに状態を初期化
            temporal = TemporalContext(max_seconds=60.0, fps=fps)
            for row in rows:
                features = _row_to_features(row)
                temporal.append(features)
                label = (row.get("label") or "").strip()
                if not label:
                    continue
                assessment = classifier.assess(_observation(features, temporal))
                y_true.append(label)
                y_pred.append(_state(assessment, min_level, awake))
    return y_true, y_pred
