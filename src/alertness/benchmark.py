"""起きている人の録画 CSV で、誤警告と見逃しをまとめて測る。

    python -m alertness.benchmark runs/ingested --config config/browser.yaml
    python -m alertness.benchmark runs/rehearsal --no-inject

しきい値や cue を変えたら、これで前後を比べる。誤警告だけを減らす変更は簡単で、
見逃しが増えていないことを同時に確かめないと改善かどうか分からない。
入力は「起きていて普通にしている」区間の収録であること（警告はすべて誤警告として数える）。
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
from collections import defaultdict
from multiprocessing import Pool

from .config import load_config
from .evaluation import benchmark
from .evaluation.scenarios import SCENARIOS


def _job(args: tuple[str, str, int, bool]) -> dict:
    path, config_path, step, inject = args
    config = load_config(config_path)
    frames = benchmark.load_features(path, step)
    control = list(benchmark.replay(frames, config))
    result = {"path": path, "false_alarms": benchmark.false_alarms(control, config)}
    duration = control[-1].t if control else 0.0
    if inject:
        result["detection"] = {
            s.name: benchmark.detection(frames, config, s, control)
            for s in SCENARIOS
            if duration >= benchmark.INJECT_AT + s.seconds + 5.0
        }
    return result


def _paths(items: list[str]) -> list[str]:
    found: list[str] = []
    for item in items:
        if os.path.isdir(item):
            found += sorted(glob.glob(os.path.join(item, "**", "*.csv"), recursive=True))
        else:
            found += sorted(p for p in glob.glob(item) if os.path.isfile(p))
    return found


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("1 以上を指定してください")
    return value


def _print_false_alarms(results: list[dict]) -> None:
    print(f"誤警告（開始 {benchmark.WARMUP_SECONDS:.0f} 秒以降・{len(results)} 本の平均）")
    dims = sorted({d for r in results for d in r["false_alarms"]["alert_share"]})
    for dim in dims:
        share = statistics.mean(r["false_alarms"]["alert_share"].get(dim, 0.0) for r in results)
        per_hour = statistics.mean(
            r["false_alarms"]["episodes_per_hour"].get(dim, 0.0) for r in results
        )
        causes: defaultdict[str, float] = defaultdict(float)
        for r in results:
            for cue, s in r["false_alarms"]["causes"].get(dim, {}).items():
                causes[cue] += s / len(results)
        top = "、".join(f"{c} {v:.1%}" for c, v in sorted(causes.items(), key=lambda x: -x[1])[:4])
        startup = statistics.mean(
            r["false_alarms"]["startup_share"].get(dim, 0.0) for r in results
        )
        print(f"  {dim:14} 時間 {share:6.1%}  {per_hour:5.0f} 回/時  "
              f"開始直後 {startup:6.1%}  原因: {top}")


def _print_detection(results: list[dict]) -> None:
    print("差し込んだ状態（検出率は差し込み区間で MEDIUM 以上になった収録の割合）")
    print(f"  {'状態':18}{'軸':14}{'検出':>6}{'差込なし':>8}{'遅れ中央':>8}{'遅れ90%':>8}")
    for scenario in SCENARIOS:
        rows = [r["detection"][scenario.name] for r in results if scenario.name in r["detection"]]
        if not rows:
            continue
        hit = sum(row["injected"]["latency"] is not None for row in rows) / len(rows)
        base = sum(row["control"]["latency"] is not None for row in rows) / len(rows)
        latencies = sorted(
            row["injected"]["latency"]
            for row in rows
            if row["injected"]["latency"] is not None and not row["injected"]["alerting_before"]
        )
        median = f"{statistics.median(latencies):.1f}s" if latencies else "-"
        p90 = f"{latencies[int(0.9 * (len(latencies) - 1))]:.1f}s" if latencies else "-"
        label = scenario.name if scenario.expect_alert else f"{scenario.name}（上がらないのが正）"
        print(f"  {label:18}{scenario.dimension:14}{hit:6.0%}{base:8.0%}{median:>8}{p90:>8}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="起きている人の録画 CSV かフォルダ（再帰的に探す）")
    parser.add_argument("--config", default="config/browser.yaml")
    parser.add_argument("--step", type=_positive, default=2,
                        help="何行おきに読むか。35fps の収録を 2 で端末経由の 17fps 前後に寄せる")
    parser.add_argument("--no-inject", action="store_true", help="誤警告だけを測る")
    parser.add_argument("--jobs", type=_positive, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)

    paths = _paths(args.csv)
    if not paths:
        print("CSV が見つかりません。")
        return 1
    jobs = [(p, args.config, args.step, not args.no_inject) for p in paths]
    with Pool(min(args.jobs, len(jobs))) as pool:
        results = pool.map(_job, jobs, chunksize=1)
    _print_false_alarms(results)
    if not args.no_inject:
        _print_detection(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
