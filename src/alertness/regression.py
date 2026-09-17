"""実機で録った収録を流し直し、書いておいた期待どおりに判定されるかを確かめる。

    python -m alertness.regression runs/regression/phone.json
    python -m alertness.regression runs/regression/phone.json --config config/browser.yaml

判定の中身（cue・しきい値・設定）を変えたら、スマホで試し直す前にこれを通す。
カメラの読み取りや通信を変えたときは、流し直しでは確かめられないので実機で試す。
期待の書き方は alertness.evaluation.regression を参照。
"""

from __future__ import annotations

import argparse

from .config import load_config
from .evaluation import regression


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="期待を書いた JSON")
    parser.add_argument("--config", default="config/browser.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    failed = 0
    total = 0
    for csv_path, expectations in regression.load(args.spec):
        print(csv_path)
        for outcome in regression.check_recording(csv_path, expectations, config):
            total += 1
            failed += not outcome.passed
            e = outcome.expectation
            when = f"{e['at']:.1f}s" if "at" in e else f"{e['from']:.1f}〜{e['to']:.1f}s"
            mark = "OK  " if outcome.passed else "NG  "
            detail = "" if outcome.passed else f"  → {outcome.observed}"
            what = f"{e['kind']:8} {e['dimension']:13} {e.get('note', '')}"
            print(f"  {mark}{when:>13} {what}{detail}")
    print(f"{total - failed}/{total} 件が期待どおり")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
