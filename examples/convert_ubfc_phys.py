"""UBFC-Phys → manifest 変換器。実験プロトコル(TSST)からストレスの教師ラベルを付ける。

配布形式:
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

## アノテ規約: 実験条件から写す

ラベルは実験条件そのものから写す。UBFC-Phys は TSST に準拠した設計で、
T1=安静 / T2=スピーチ / T3=暗算 という誘発条件自体がデータセットの根拠になる。

難易度も段階に反映する。このデータセットは難易度2水準(test/ctrl)を被験者へランダムに
割り当てており、同じタスクでも掛かる負荷が違う。段階を難易度で分けると4段階すべてが
埋まり、ctrl の被験者も捨てずに済む。

生体信号(BVP)の HRV から段階を出す経路は取らない。E4 は手首装着なので、発話や体動の
ある区間では腕の動きを拾って拍検出が崩れ、RMSSD が生理的にありえない桁まで跳ねる。
--bvp-report で窓ごとの拍数・心拍・RMSSD を見れば、どの区間が使えないか確認できる。

## 被験者ゲート: 誘発が効いた被験者だけを使う

実験条件からラベルを写す以上、ストレスが実際に掛かっていない被験者を入れるとラベルが
嘘になる。selfReportedAnx は CSAI-2 の3尺度（認知不安・身体不安・自信）を実験前後で
測ったもので、認知不安と身体不安は上がるほど、自信は下がるほど負荷が掛かったことを示す。

3つが揃って動くとは限らず、片方だけ動く被験者や、不安が上がりながら自信も上がる被験者が
いる。1尺度だけで判定すると取りこぼすので、3つを合算した値で見る（induction_score）。

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

# アノテ規約: シナリオ(難易度) × タスク → ストレスの段階。
# どの難易度でも T2(スピーチ)を T3(暗算)より上に置く。評価者の前での公開スピーチは
# TSST の中核的ストレッサで、暗算より強い反応を出すのが定説のため。
# ctrl は難易度が低い側なので、同じタスクでも一段軽い段階に写す。
# ここを変えるときは docs/annotation-guide.md のストレスの表と必ず揃えること。
TASK_STAGE = {
    "test": {"T1": "none", "T2": "high", "T3": "medium"},
    "ctrl": {"T1": "none", "T2": "medium", "T3": "low"},
}

# 誘発が効いたと見なす自己申告スコアの下限。0 は「不安の上昇か自信の低下が差し引きで
# 残っている」という最低限の線。厳しくするほど被験者は減るが、ラベルの確度は上がる。
INDUCTION_MIN_SCORE = 0.0

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


def induction_score(anxiety: dict[str, tuple[float, float]]) -> float:
    """自己申告(CSAI-2)の実験前後の変化を1つの値にする。正なら誘発が効いた向き。

    認知不安・身体不安は上がるほど、自信は下がるほど負荷が掛かったことを示すので、
    自信だけ符号を反転して足す。
    """
    def change(key: str) -> float:
        pre, post = anxiety[key]
        return post - pre

    return change("cognitive") + change("somatic") - change("confidence")


def induction_worked(info: dict[str, str], anxiety: dict[str, tuple[float, float]]) -> str:
    """ストレス誘発が効いた被験者か。効いていなければ理由を返す（効いていれば空文字）。"""
    scenario = info["scenario"].lower()
    if scenario not in TASK_STAGE:
        return f"未知のシナリオ: {info['scenario']}（{sorted(TASK_STAGE)} のいずれかであること）"
    score = induction_score(anxiety)
    if score <= INDUCTION_MIN_SCORE:
        return f"自己申告に誘発の跡がない（合成スコア {score:+.2f}）"
    return ""


def convert_subject(root: Path, subject: str, out_dir: Path, *, force: bool) -> list[Path]:
    """1被験者の3タスクを manifest にする。書き出したパスを返す。"""
    subject_dir = root / subject
    info = read_info(subject_dir / f"info_{subject}.txt")
    anxiety = read_anxiety(subject_dir / f"selfReportedAnx_{subject}.csv")
    print(
        f"{subject} (scenario={info['scenario']}, "
        f"自己申告スコア {induction_score(anxiety):+.2f})"
    )

    reason = induction_worked(info, anxiety)
    if reason and not force:
        print(f"  スキップ: {reason}。--force で無視できます。")
        return []
    if reason:
        print(f"  ⚠ {reason} が、--force により続行します。")

    stages = TASK_STAGE[info["scenario"].lower()]
    written = []
    for task in TASKS:
        # 区間の長さは BVP の標本数から出す（動画を開かずに秒数が分かる）。
        duration = read_signal(subject_dir / f"bvp_{subject}_{task}.csv").size / FS
        path = write_manifest(
            out_dir / f"ubfc_{subject}_{task}.json",
            video=(subject_dir / f"vid_{subject}_{task}.avi").as_posix(),
            subject=subject,
            context=CONTEXT,
            segments=[segment(0.0, duration, stress=stages[task])],
        )
        print(f"  {task}: stress={stages[task]} (0.0〜{duration:.1f}s) → {path}")
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
        print(f"{root} に被験者フォルダ(s1 等)がありません。展開しましたか？", file=sys.stderr)
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
