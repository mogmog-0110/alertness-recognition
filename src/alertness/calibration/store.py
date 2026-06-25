"""キャリブ結果（プロファイル）の保存。

プロファイルは個人データなので、保存先（profiles/）はリポジトリに含めない。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts import CalibrationProfile


def save_profile(profile: CalibrationProfile, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "ear_open_baseline": profile.ear_open_baseline,
        "mar_neutral": profile.mar_neutral,
        "head_pose_neutral": {
            "pitch": profile.head_pose_neutral.pitch,
            "yaw": profile.head_pose_neutral.yaw,
            "roll": profile.head_pose_neutral.roll,
        },
        "gaze_center": list(profile.gaze_center),
        "face_scale": profile.face_scale,
        "user_id": profile.user_id,
        "schema_version": profile.schema_version,
    }
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
