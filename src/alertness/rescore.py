"""録画CSVを、いまの設定で判定し直す。

収録し直さずに設定を試せる。CSV には特徴量が全部残っているので、判定だけを
やり直せばよい。evaluate は記録済みの dim_*_level 列を読むだけなので、
しきい値を変えるたびに 5 分の収録が要る形になっていた。

例:
    python -m alertness.rescore runs/session_1.csv --config config/default.yaml
    python -m alertness.rescore runs/ -o rescored/

出力は入力と同じ形式（dim_* 列だけ入れ替え）なので、そのまま evaluate に渡せる。
"""

from __future__ import annotations

import argparse
import csv
import os

from . import factory
from .config import load_config
from .contracts import CalibrationProfile, Features, Frame, Observation
from .evaluation.paths import resolve_csv_paths
from .temporal import TemporalContext

# 判定に使わない列。ここ以外はすべて特徴量として読み込む。
_NOT_FEATURES = frozenset({"label", "subject", "session", "source_id", "frame_index"})


def _features_from(row: dict[str, str]) -> Features | None:
    values: dict[str, float] = {}
    timestamp = None
    for key, raw in row.items():
        if key in _NOT_FEATURES or key.startswith(("dim_", "cue_")):
            continue
        if raw == "" or raw is None:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value != value:  # NaN
            continue
        if key == "timestamp":
            timestamp = value
        else:
            values[key] = value
    if timestamp is None:
        return None
    present = row.get("face_present", "").lower() in ("1", "1.0", "true")
    return Features(values=values, timestamp=timestamp, face_present=present)


def rescore_file(path: str, config: dict, out_path: str) -> int:
    classifier = factory.build_classifier(config)
    fps = config.get("camera", {}).get("target_fps", 30) or 30
    temporal = TemporalContext(max_seconds=60.0, fps=float(fps))
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 0

    # 元の列順を保つ。dim_* だけ差し替えるので、evaluate がそのまま読める。
    fieldnames = list(rows[0].keys())
    written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            features = _features_from(row)
            if features is None:
                continue
            temporal.append(features)
            frame = Frame(image=None, index=written, timestamp=features.timestamp)
            # landmarks は判定に使わない (cue は features と history だけ見る)。
            # profile は正規化済みの値を読むので同一で素通しにする。
            obs = Observation(
                frame=frame, landmarks=None, features=features,
                history=temporal, profile=CalibrationProfile.identity(),
            )
            assessment = classifier.assess(obs)
            out = dict(row)
            for name, dim in assessment.dimensions.items():
                if f"dim_{name}_score" in out:
                    out[f"dim_{name}_score"] = f"{dim.score:.6f}"
                    out[f"dim_{name}_level"] = str(int(dim.level))
            for cue in assessment.cues:
                if f"cue_{cue.name}" in out:
                    out[f"cue_{cue.name}"] = f"{cue.score:.6f}"
            writer.writerow(out)
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*", help="判定し直すCSV/フォルダ")
    parser.add_argument("--config", default="config/default.yaml", help="設定ファイル")
    parser.add_argument("-o", "--out", default="rescored", help="出力先フォルダ")
    args = parser.parse_args(argv)

    paths = resolve_csv_paths(args.csv)
    if not paths:
        print("CSVが見つかりません。")
        return 1
    config = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    for path in paths:
        out_path = os.path.join(args.out, os.path.basename(path))
        count = rescore_file(path, config, out_path)
        print(f"{os.path.basename(path)}: {count} 行 -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
