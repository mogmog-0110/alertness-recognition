"""デモ用の起動。引数なしで動く。

    python -m alertness.demo

証明書の作成もアドレスの検出も自分でやる。人前で動かすときに、IP を調べて
コマンドに渡して……という手順を挟まずに済ませるためのもの。

設定を変えたいときは通常の入口を使う:
    python -m alertness --config config/browser.yaml
"""

from __future__ import annotations

import os
import sys

from .app import main as run_app
from .webcert import local_ip

_CONFIG = os.path.join("config", "browser.yaml")


def main(argv: list[str] | None = None) -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = _CONFIG if os.path.isfile(_CONFIG) else os.path.join(root, _CONFIG)
    if not os.path.isfile(config):
        print(f"設定が見つかりません: {config}")
        return 1

    address = local_ip()
    if address.startswith("127."):
        print("この PC の LAN アドレスが取れません。Wi-Fi に繋がっているか確認してください。")
        return 1

    print("=" * 56)
    print("  端末のブラウザで次を開いてください")
    print(f"      https://{address}:8765/")
    print()
    print("  1. 証明書の警告を承認する（「詳細」→「アクセスする」系）")
    print("  2. 「はじめる」→ カメラを許可")
    print("  3. 4 秒後に基準の測定が始まります。前方を見てください")
    print()
    print("  同じ Wi-Fi にいること。繋がらないときはファイアウォールで")
    print("  8765 の受信を許可してください。")
    print("=" * 56)
    print()
    # 証明書は起動時に RemoteLink が用意する（IP が変わっていれば作り直す）。
    return run_app(["--config", config, *(argv or sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
