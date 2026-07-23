"""UBFC-Phys → manifest 変換器。実験プロトコル(TSST)からストレスの教師ラベルを付ける。

配布形式（実ファイルで確認済み）:
  data/UBFC-Phys/s1/
    info_s1.txt              5行: 被験者ID / 性別(m,f) / シナリオ(test,ctrl) / 日付 / 開始時刻
    bvp_s1_T1.csv            64Hz・ヘッダ無し・1行1値。180秒=11520行
    eda_s1_T1.csv            4Hz（本変換器では使わない）
    vid_s1_T1.avi            1024x1024 / 35fps / 約3分
    selfReportedAnx_s1.csv   3行2列。行=認知不安/身体不安/自信、列=実験前/実験後

## 付ける軸は stress だけ

UBFC-Phys は社会的ストレス誘発(TSST準拠)の実験で、眠気は誘発も測定もされていない。
3分間のスピーチ・暗算タスクで眠くなる被験者はいないので、drowsiness を none と埋めるのは
嘘になる。docs/annotation-guide.md の「区間には情報のある軸だけを付ければよい。付けなかった
軸は未アノテであって none とは断定しない」に従い、stress だけを付ける。

## アノテ規約: 実験プロトコルから写す（生体信号からは写さない）

当初は BVP から RMSSD を出して段階化する方針だったが、**s1 の実データで検証したところ
使いものにならなかった**ので取りやめた。記録を残す（--bvp-report で追試できる）:

  - T2/T3 の窓RMSSDが 300〜900ms。人の安静時 20〜50ms と桁が2つ違い、生理的にありえない
  - T2(発話課題)の推定心拍が T1(安静)より低く出る（53-77bpm vs 78-86bpm）＝拍の取りこぼし
  - 原因は体動。T2 の支配周波数は 0.77Hz(=46bpm) で、心拍ではなく発話・体動の成分。
    窓ごとの振幅も std 109→23 と4.7倍振れる。E4 は手首装着なので発話中の腕の動きを拾う
  - 帯域通過・局所正規化・2推定器の一致ゲートまで試したが、窓単位のHRは安定しなかった

これは config/default.yaml が rPPG の HRV を hrv_enabled: false にしている理由
（「雑音の乗った脈波でのピーク検出そのものが崩れている」）と同じ現象が、接触PPGでも
起きたということ。bio/peaks.py の detect_peaks を頑健にするまで、HRV由来のラベルは使えない。

代わりに **実験プロトコルそのものを正解として使う**。UBFC-Phys は TSST に準拠した設計で、
T1=安静 / T2=スピーチ / T3=暗算 という誘発条件自体がデータセットの根拠なので、これは
妥当な使い方。ただし「誘発が効いた被験者に限る」という条件が付くので、下の被験者ゲートで絞る。

## 被験者ゲート: 誘発が効いた被験者だけを使う

プロトコルからラベルを写す以上、ストレスが実際に掛かっていない被験者を入れるとラベルが
嘘になる。2つの条件で絞る:

  - info の scenario が test（ctrl は難易度が低い側でストレスが上がりきらない可能性）
  - selfReportedAnx の認知不安が 実験前 < 実験後（誘発が効いた自己申告の裏づけ）

s1 は scenario=test、認知不安 2.143→3.286、自信 2.778→1.778 で両方を満たす。

## 使い方

    python examples\\convert_ubfc_phys.py                      :: data/UBFC-Phys の全被験者
    python examples\\convert_ubfc_phys.py data\\UBFC-Phys s1    :: 被験者を指定
    python examples\\convert_ubfc_phys.py --bvp-report          :: BVPの診断だけ（ラベルは書かない）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from alertness.bio import mean_hr, rmssd, rr_intervals_ms
from alertness.bio.peaks import peak_times
from alertness.ingest.mapping import segment, write_manifest

FS = 64.0  # BVP のサンプリング周波数[Hz]。Empatica E4 の仕様。
TASKS = ("T1", "T2", "T3")

# アノテ規約: タスク → ストレスの段階。
# T2(スピーチ)を最上位に置く。評価者の前での公開スピーチは TSST の中核的ストレッサで、
# 暗算より強い反応を出すのが定説のため。段階に幅を持たせる意味でも T3 を medium にする。
# ここを変えるときは docs/annotation-guide.md のストレスの表と必ず揃えること。
TASK_STAGE = {"T1": "none", "T2": "high", "T3": "medium"}

# 用途タグ。UBFC-Phys は着座の実験室環境で運転ではないので driving とは書かない。
# stress は用途非依存の軸（colab の CONTEXT_FREE_AXES）なので、空でも学習に影響しない。
CONTEXT = ""

REPORT_WINDOW_S = 30.0  # --bvp-report の窓幅[秒]


def read_signal(path: Path) -> np.ndarray:
    """1行1値・ヘッダ無しのCSVを読む。UBFC-Phys の bvp/eda はこの形式。"""
    if not path.exists():
        raise FileNotFoundError(f"信号ファイルが見つかりません: {path}")
    values = np.loadtxt(path, dtype=float, ndmin=1)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"1列の数値が並んだCSVではありません: {path}")
    return values


def read_info(path: Path) -> dict[str, str]:
    """info_sN.txt から被験者ID・性別・シナリオ(test/ctrl)を読む。"""
    if not path.exists():
        raise FileNotFoundError(f"info が見つかりません: {path}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"info の行が足りません（ID/性別/シナリオが必要）: {path}")
    return {"subject": lines[0], "gender": lines[1], "scenario": lines[2]}


def read_anxiety(path: Path) -> dict[str, tuple[float, float]]:
    """selfReportedAnx_sN.csv を読む。3行2列（行=指標、列=実験前,実験後）。"""
    if not path.exists():
        raise FileNotFoundError(f"自己申告スコアが見つかりません: {path}")
    table = np.loadtxt(path, delimiter=",", dtype=float, ndmin=2)
    if table.shape != (3, 2):
        raise ValueError(f"3行2列ではありません（{table.shape}）: {path}")
    names = ("cognitive", "somatic", "confidence")
    return {name: (float(row[0]), float(row[1])) for name, row in zip(names, table, strict=True)}


def induction_worked(info: dict[str, str], anxiety: dict[str, tuple[float, float]]) -> str:
    """ストレス誘発が効いた被験者か。効いていなければ理由を返す（効いていれば空文字）。"""
    if info["scenario"].lower() != "test":
        return f"scenario={info['scenario']}（ctrl は誘発が弱い側）"
    pre, post = anxiety["cognitive"]
    if post <= pre:
        return f"認知不安が上がっていない（{pre:.3f} → {post:.3f}）"
    return ""


def convert_subject(root: Path, subject: str, out_dir: Path, *, force: bool) -> list[Path]:
    """1被験者の3タスクを manifest にする。書き出したパスを返す。"""
    subject_dir = root / subject
    info = read_info(subject_dir / f"info_{subject}.txt")
    anxiety = read_anxiety(subject_dir / f"selfReportedAnx_{subject}.csv")
    pre, post = anxiety["cognitive"]
    print(f"{subject} (scenario={info['scenario']}, 認知不安 {pre:.3f}→{post:.3f})")

    reason = induction_worked(info, anxiety)
    if reason and not force:
        print(f"  スキップ: {reason}。--force で無視できます。")
        return []
    if reason:
        print(f"  ⚠ {reason} が、--force により続行します。")

    written = []
    for task in TASKS:
        # 区間の長さは BVP の標本数から出す（動画を開かずに秒数が分かる）。
        duration = read_signal(subject_dir / f"bvp_{subject}_{task}.csv").size / FS
        path = write_manifest(
            out_dir / f"ubfc_{subject}_{task}.json",
            video=(subject_dir / f"vid_{subject}_{task}.avi").as_posix(),
            subject=subject,
            context=CONTEXT,
            segments=[segment(0.0, duration, stress=TASK_STAGE[task])],
        )
        print(f"  {task}: stress={TASK_STAGE[task]} (0.0〜{duration:.1f}s) → {path}")
        written.append(path)
    return written


def report_bvp(root: Path, subject: str) -> None:
    """BVPの診断。ラベルは書かない。HRV由来ラベルを再検討するときの土台。"""
    size = int(round(REPORT_WINDOW_S * FS))
    print(f"{subject}: 窓={REPORT_WINDOW_S:.0f}秒  （人の安静時 RMSSD は 20〜50ms が目安）")
    for task in TASKS:
        signal = read_signal(root / subject / f"bvp_{subject}_{task}.csv")
        spectrum = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
        freqs = np.fft.rfftfreq(signal.size, 1.0 / FS)
        band = (freqs >= 0.6) & (freqs <= 3.5)
        print(f"  {task}: 支配周波数={freqs[band][np.argmax(spectrum[band])] * 60:.0f}bpm")
        for i in range(0, signal.size - size + 1, size):
            rr = rr_intervals_ms(peak_times(signal[i : i + size], FS))
            print(
                f"    {i / FS:5.0f}s  拍数={rr.size + 1:3d}  HR={mean_hr(rr):6.1f}bpm  "
                f"RMSSD={rmssd(rr):7.1f}ms  振幅std={signal[i : i + size].std():6.1f}"
            )


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a for a in argv[1:] if a.startswith("--")}
    root = Path(args[0] if args else "data/UBFC-Phys")
    if not root.is_dir():
        print(f"データセットのフォルダが見つかりません: {root}", file=sys.stderr)
        return 1
    subjects = args[1:] or [p.name for p in sorted(root.glob("s*")) if p.is_dir()]
    if not subjects:
        print(f"{root} に被験者フォルダ(s1 等)がありません。zip を展開しましたか？", file=sys.stderr)
        return 1

    if "--bvp-report" in flags:
        for subject in subjects:
            report_bvp(root, subject)
        return 0

    out_dir = Path("data/manifests")
    for subject in subjects:
        convert_subject(root, subject, out_dir, force="--force" in flags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
