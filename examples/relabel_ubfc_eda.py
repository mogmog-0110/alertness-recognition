"""取り込み済みCSVに、EDA由来のストレスラベル列(label_stress_eda)を足す。

実験プロトコルから写した label_stress は、負荷が掛かった「はず」の仮定で、被験者の
実際の反応とはよくズレる（心拍で見ると18人中5人しか一致しない）。EDA(皮膚電気活動)は
体動に強く、UBFC-Phys では18人中16人がタスクで上がった。EDA から窓単位の覚醒度を出し、
被験者ごとに安静基準で相対化してラベルにすると、被験者独立の2値ストレス判定が
macro-F1 0.72→0.79、見逃しも4割減った。

映像は捨ててあるが EDA(eda_sN_TX.csv)と特徴量CSVは残っているので、再取り込みは要らない。
既存CSVに列を足すだけ（label_stress は残す）。何度実行してもよい。

## 使い方

    python examples\\relabel_ubfc_eda.py                     :: runs/ingested を全部ラベル付け
    python examples\\relabel_ubfc_eda.py --out runs/ingested :: 出力先（既定は上書き）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from alertness.bio import relative_arousal, stage_from_arousal, subject_scale, tonic_windows

EDA_FS = 4.0  # Empatica E4 の EDA サンプリング周波数[Hz]
WINDOW_S = 20.0  # ラベルを付ける窓[秒]。1タスク180秒が9窓になる。
TASKS = ("T1", "T2", "T3")
# 覚醒度(0..1)→段階の昇順しきい値。2値(落ち着き/上昇)に束ねて使う前提。境界は low/medium 間。
AROUSAL_THRESHOLDS = (0.15, 0.40, 0.70)
LABEL_COLUMN = "label_stress_eda"


def _eda_windows(data_root: Path, subject: str, task: str) -> list[tuple[float, float]]:
    path = data_root / subject / f"eda_{subject}_{task}.csv"
    if not path.exists():
        return []
    return tonic_windows(np.loadtxt(path), EDA_FS, WINDOW_S)


def _subject_scale(data_root: Path, subject: str):
    rest = _eda_windows(data_root, subject, "T1")
    every = [w for task in TASKS for w in _eda_windows(data_root, subject, task)]
    return subject_scale(rest, every)


def label_clip(df: pd.DataFrame, windows: list[tuple[float, float]], scale) -> list[str] | None:
    """フレームごとに、その時刻の窓の覚醒度から段階を割り当てる。"""
    if not windows or scale is None:
        return None
    baseline, spread = scale
    centers = np.array([c for c, _ in windows])
    stages = [
        stage_from_arousal(relative_arousal(scl, baseline, spread), AROUSAL_THRESHOLDS)
        for _, scl in windows
    ]
    idx = np.clip(np.searchsorted(centers, df["timestamp"].to_numpy()), 0, len(stages) - 1)
    return [stages[i] for i in idx]


def _task_of(clip_dir: Path) -> str:
    # runs/ingested/s9__vid_s9_T2 → T2
    return clip_dir.name.rsplit("_", 1)[-1]


def relabel(ingested: Path, data_root: Path, out_base: Path) -> tuple[int, int]:
    """取り込み済みCSVにEDAラベル列を足す。(付けたクリップ数, 飛ばした数) を返す。"""
    scales: dict[str, object] = {}
    labeled = skipped = 0
    for clip_dir in sorted(ingested.glob("s*__*")):
        subject = clip_dir.name.split("__")[0]
        task = _task_of(clip_dir)
        csvs = list(clip_dir.glob("*.csv"))
        if not csvs:
            continue
        if subject not in scales:
            scales[subject] = _subject_scale(data_root, subject)
        stages = label_clip(
            pd.read_csv(csvs[0]), _eda_windows(data_root, subject, task), scales[subject]
        )
        if stages is None:
            print(f"{clip_dir.name}: EDA が無く付けられません。飛ばす", file=sys.stderr)
            skipped += 1
            continue
        df = pd.read_csv(csvs[0])
        df[LABEL_COLUMN] = stages
        out_dir = out_base / clip_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / csvs[0].name, index=False)
        labeled += 1
    return labeled, skipped


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="取り込み済みCSVにEDA由来ラベルを足す")
    parser.add_argument("--ingested", default="runs/ingested", help="取り込み済みCSVの場所")
    parser.add_argument("--data", default="data/UBFC-Phys", help="eda_*.csv のある被験者フォルダ")
    parser.add_argument("--out", default="runs/ingested", help="出力先（既定は上書き）")
    args = parser.parse_args(argv)

    ingested = Path(args.ingested)
    if not ingested.is_dir():
        print(f"取り込み済みフォルダがありません: {ingested}", file=sys.stderr)
        return 1
    labeled, skipped = relabel(ingested, Path(args.data), Path(args.out))
    print(f"EDAラベルを付けたクリップ: {labeled} / 飛ばし: {skipped}")
    print(f"学習では target='{LABEL_COLUMN}', label_collapse='binary' を指定して使う。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
