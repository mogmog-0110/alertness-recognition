"""取り込み済みCSVに、EDA由来のストレスラベル列(label_stress_eda)を足す。

実験プロトコルから写した label_stress は、負荷が掛かった「はず」の仮定で、被験者の
実際の反応とはズレることがある。EDA(皮膚電気活動)は体動に強く、実際の生理反応を
拾える。EDA から窓単位の覚醒度を出し、被験者ごとに安静基準で相対化してラベルにする。

映像を捨てても EDA(eda_sN_TX.csv)と特徴量CSVがあれば、再取り込み無しで付けられる。
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

from alertness.bio import (
    relative_arousal,
    stage_from_arousal,
    stress_rise,
    subject_scale,
    tonic_windows,
)

EDA_FS = 4.0  # Empatica E4 の EDA サンプリング周波数[Hz]
WINDOW_S = 20.0  # ラベルを付ける窓[秒]。1タスク180秒が9窓になる。
TASKS = ("T1", "T2", "T3")
# 覚醒度(0..1)→段階の昇順しきい値。2値(落ち着き/上昇)に束ねて使う前提。境界は low/medium 間。
AROUSAL_THRESHOLDS = (0.15, 0.40, 0.70)
# EDAゲート: ストレス時の上昇がこの値未満の被験者は、EDAが反応していない＝ラベルの根拠が
# 無いので外す。反応者と非反応者はこの値を挟んで分かれる。
MIN_STRESS_RISE = 0.2
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


def _responder(data_root: Path, subject: str) -> tuple[bool, float | None]:
    """EDAがストレス時に上がった被験者か。(反応したか, 上昇量) を返す。"""
    rest = _eda_windows(data_root, subject, "T1")
    stress = [w for task in ("T2", "T3") for w in _eda_windows(data_root, subject, task)]
    every = [w for task in TASKS for w in _eda_windows(data_root, subject, task)]
    rise = stress_rise(rest, stress, every)
    return (rise is not None and rise >= MIN_STRESS_RISE), rise


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


def relabel_subject(
    subject: str, ingested: Path, data_root: Path, *, gate: bool = True
) -> tuple[int, str | None]:
    """1被験者の取り込み済みクリップにEDAラベル列を上書きで足す。

    その被験者の3タスクが揃ってから呼ぶ（安静基準と振れ幅に全タスクが要る）。
    (付けたクリップ数, 除外理由 or None) を返す。取り込みパイプラインから呼ぶ用。
    """
    if gate:
        ok, rise = _responder(data_root, subject)
        if not ok:
            return 0, "EDA無し" if rise is None else f"EDA非反応（上昇 {rise:+.2f}）"
    scale = _subject_scale(data_root, subject)
    labeled = 0
    for clip_dir in sorted(ingested.glob(f"{subject}__*")):
        csvs = list(clip_dir.glob("*.csv"))
        if not csvs:
            continue
        stages = label_clip(
            pd.read_csv(csvs[0]), _eda_windows(data_root, subject, _task_of(clip_dir)), scale
        )
        if stages is None:
            continue
        df = pd.read_csv(csvs[0])
        df[LABEL_COLUMN] = stages
        df.to_csv(csvs[0], index=False)  # 同じCSVに列を足して上書き
        labeled += 1
    return labeled, None


def relabel(
    ingested: Path, data_root: Path, out_base: Path, *, gate: bool = True
) -> tuple[int, int]:
    """取り込み済みCSVにEDAラベル列を足す。(付けたクリップ数, 飛ばした数) を返す。

    gate=True なら、EDAがストレス時に上がらなかった非反応者を外す（ラベルの根拠が無いため）。
    """
    scales: dict[str, object] = {}
    responders: dict[str, bool] = {}
    labeled = skipped = 0
    for clip_dir in sorted(ingested.glob("s*__*")):
        subject = clip_dir.name.split("__")[0]
        task = _task_of(clip_dir)
        csvs = list(clip_dir.glob("*.csv"))
        if not csvs:
            continue
        if subject not in responders:
            ok, rise = _responder(data_root, subject)
            responders[subject] = ok
            if gate and not ok:
                rise_str = "EDA無し" if rise is None else f"上昇 {rise:+.2f}"
                print(f"{subject}: EDA非反応（{rise_str}）。除外", file=sys.stderr)
        if gate and not responders[subject]:
            skipped += 1
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
    parser.add_argument("--no-gate", action="store_true", help="EDA非反応者も除外せず付ける")
    args = parser.parse_args(argv)

    ingested = Path(args.ingested)
    if not ingested.is_dir():
        print(f"取り込み済みフォルダがありません: {ingested}", file=sys.stderr)
        return 1
    labeled, skipped = relabel(ingested, Path(args.data), Path(args.out), gate=not args.no_gate)
    print(f"EDAラベルを付けたクリップ: {labeled} / 飛ばし: {skipped}")
    print(f"学習では target='{LABEL_COLUMN}', label_collapse='binary' を指定して使う。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
