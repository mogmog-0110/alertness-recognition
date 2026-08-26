"""特徴量と判定結果をCSVに書き出す出力先。後の学習(Colab)の入力になる。

列は固定（FEATURE_COLUMNS）にしてある。フレームごとに値が欠けても列は変えず、
学習時と推論時で同じ並びになるようにする。これが学習との契約になる。
"""

from __future__ import annotations

import csv
import time
from collections.abc import Sequence
from pathlib import Path

from ..contracts import Assessment, Observation
from ..labeling import LabelState

# 記録する特徴量の列（生 + 正規化）。順序と名前が学習との取り決め。
FEATURE_COLUMNS = (
    "ear",
    "ear_left",
    "ear_right",
    "mar",
    "pitch",
    "yaw",
    "roll",
    "gaze_x",
    "face_scale",
    "ear_norm",
    "mar_rel",
    "pitch_rel",
    "yaw_rel",
    "gaze_off",
    "gaze_dx",  # 向き付きの視線ズレ。左右のどちらを見たかは gaze_off では分からない
    "normalize_version",
    "jawOpen",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    # FACS の AU に対応する blendshape（AU4/AU7/AU24/AU15/AU6）。判定にも学習にも使う。
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "eyeSquintLeft",
    "eyeSquintRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "cheekSquintLeft",
    "cheekSquintRight",
    "hr_bpm",  # rPPG 由来（無効時は空）。stress モデルの特徴になり得る。
    "rppg_quality",
    "hrv_rmssd",  # rPPG 由来。良質な窓でだけ入る（機会的）ので欠けるのが普通。
    "resp_rpm",  # rPPG 由来の呼吸数。窓(30秒)が埋まるまで空。
    "resp_quality",
)


class CsvRecorderSink:
    def __init__(
        self,
        path_dir: str,
        dimension_names: Sequence[str],
        labels: LabelState | None = None,
        subject: str = "",
        cue_names: Sequence[str] = (),
        context: str = "",
    ) -> None:
        directory = Path(path_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._path = directory / time.strftime("session_%Y%m%d_%H%M%S.csv")
        self._file = self._path.open("w", newline="", encoding="utf-8")
        self._labels = labels or LabelState()  # 実行時に書き換わる正解ラベル
        self._subject = subject  # 被験者ID（被験者独立の評価に使う）
        self._context = context  # 用途タグ（driving/study 等。学習で絞る/プールに使う）
        self._session_id = self._path.stem

        # session_id / subject を残すと、後で「人ごとに分けた評価」ができる。
        fields = ["session_id", "subject", "timestamp", "face_present", *FEATURE_COLUMNS]
        for name in dimension_names:
            fields.append(f"dim_{name}_score")
            fields.append(f"dim_{name}_level")
        # cue ごとのスコアも残す。どの手がかりが効いた/誤発火したかを後で調べられる。
        for name in cue_names:
            fields.append(f"cue_{name}")
        # label は旧・単一ラベル（ライブ収録の互換）。正準は軸別の段階ラベル label_<軸>。
        fields.append("label")
        self._label_columns = [f"label_{name}" for name in dimension_names]
        fields.extend(self._label_columns)
        fields.append("context")
        # 余分なキーは無視し、欠けた列は空にする（列構成を固定するため）。
        self._writer = csv.DictWriter(
            self._file, fieldnames=fields, extrasaction="ignore", restval=""
        )
        self._writer.writeheader()

    def emit(self, obs: Observation, assessment: Assessment) -> None:
        row: dict[str, object] = dict(obs.features.values)
        row["timestamp"] = obs.features.timestamp
        row["face_present"] = int(obs.features.face_present)
        for dim in assessment.dimensions.values():
            row[f"dim_{dim.name}_score"] = dim.score
            row[f"dim_{dim.name}_level"] = int(dim.level)
        for cue in assessment.cues:
            row[f"cue_{cue.name}"] = cue.score
        row["session_id"] = self._session_id
        row["subject"] = self._subject
        row["label"] = self._labels.value
        # 軸別ラベルは、対応する provider だけが持つ（levels に無い軸は空＝未アノテ）。
        levels = self._labels.levels
        for column in self._label_columns:
            axis = column[len("label_") :]
            row[column] = levels.get(axis, "")
        row["context"] = self._context
        self._writer.writerow(row)

    def close(self) -> None:
        self._file.close()
