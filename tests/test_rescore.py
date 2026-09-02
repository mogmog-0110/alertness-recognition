"""録画CSVの判定し直しのテスト。

しきい値を変えるたびに収録し直すのは現実的でない。CSV には特徴量が全部
残っているので、判定だけやり直せる。
"""

from __future__ import annotations

import csv

from alertness.config import load_config
from alertness.rescore import rescore_file


def _write(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sample_rows(count: int = 120):
    rows = []
    for i in range(count):
        rows.append(
            {
                "timestamp": f"{i / 30:.4f}",
                "face_present": "1",
                "ear_norm": "1.0",
                "ear": "0.30",
                "yaw_rel": "0.0",
                "pitch_rel": "0.0",
                "label": "awake",
                "dim_drowsiness_score": "9.9",   # ありえない値を入れておき、
                "dim_drowsiness_level": "3",     # 上書きされることを確かめる
                "cue_eye_closure": "9.9",
            }
        )
    return rows


def test_rescoring_replaces_the_recorded_judgement(tmp_path):
    src = tmp_path / "session.csv"
    rows = _sample_rows()
    _write(src, rows, list(rows[0].keys()))
    out = tmp_path / "out.csv"

    written = rescore_file(str(src), load_config("config/default.yaml"), str(out))

    assert written == len(rows)
    got = list(csv.DictReader(open(out, encoding="utf-8")))
    # 目は開いたままなので、眠気は最大のはずがない。
    assert float(got[-1]["dim_drowsiness_score"]) < 9.0
    assert float(got[-1]["cue_eye_closure"]) < 9.0


def test_rescoring_keeps_the_label_and_columns(tmp_path):
    # 採点はラベル列を読む。列の並びも保つので、そのまま evaluate に渡せる。
    src = tmp_path / "session.csv"
    rows = _sample_rows(40)
    _write(src, rows, list(rows[0].keys()))
    out = tmp_path / "out.csv"

    rescore_file(str(src), load_config("config/default.yaml"), str(out))

    got = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [r["label"] for r in got] == ["awake"] * len(rows)
    assert list(got[0].keys()) == list(rows[0].keys())
