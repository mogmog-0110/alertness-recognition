"""デモ用の起動。引数なしで動く。

    python -m alertness.demo            # 案内だけを出す
    python -m alertness.demo --verbose  # 接続の出入りなども出す

証明書の作成もアドレスの検出も自分でやる。人前で動かすときに、IP を調べて
コマンドに渡して……という手順を挟まずに済ませるためのもの。案内の後は、人が
対応すべき異常が起きない限り何も出さない。

検出した IP が端末から届かないときは、環境変数 ALERTNESS_HOST で指定する。
設定を変えたいときは通常の入口を使う:
    python -m alertness --config config/browser.yaml
"""

from __future__ import annotations

import os
import sys
from typing import Any

from .app import parse_args, run
from .config import load_config
from .webcert import certificate_hosts, ensure, lan_addresses

_CONFIG = os.path.join("config", "browser.yaml")
_RULE = "=" * 60


def main(argv: list[str] | None = None) -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.isfile(_CONFIG):
        # 設定の中の web・models・certs は相対パスなので、置き場所ごと移る。
        os.chdir(root)
    if not os.path.isfile(_CONFIG):
        print(f"設定が見つかりません: {os.path.join(root, _CONFIG)}")
        return 1

    args = parse_args(["--config", _CONFIG, *(sys.argv[1:] if argv is None else argv)])
    config = load_config(args.config)
    net = config.get("source", {}).get("remote", {})
    try:
        address, renewed = ensure(
            net.get("certfile", "certs/cert.pem"),
            net.get("keyfile", "certs/key.pem"),
            net.get("advertise_host", ""),
        )
    except ImportError:
        print("cryptography が入っていません。scripts\\setup.bat を実行してください。")
        return 1
    if address.startswith("127."):
        print("この PC の LAN アドレスが取れません。Wi-Fi に繋がっているか確認してください。")
        return 1

    port = int(net.get("port", 8765))
    # 証明書に入っていないアドレスは案内しない。開けても証明書の不一致で繋がらない。
    covered = certificate_hosts(net.get("certfile", "certs/cert.pem"))
    others = [a for a in lan_addresses() if a != address and a in covered]
    max_peers = int(net.get("max_peers", 1))
    print(_banner(address, port, renewed, others, max_peers))
    return run(args, _without_link_notice(config))


def _banner(address: str, port: int, renewed: bool, others: list[str], max_peers: int = 1) -> str:
    lines = [
        _RULE,
        "  PC 画面の QR を端末のカメラで読み取ってください",
    ]
    if max_peers > 1:
        lines.append(f"  （同じ QR で最大 {max_peers} 台まで同時に使えます）")
    lines.append(f"  （QR が出ないときは次を開く） https://{address}:{port}/")
    if others:
        lines.append("  開けないときは: " + "  ".join(f"https://{a}:{port}/" for a in others))
    if renewed:
        lines.append("  ※ 証明書を作り直しました。端末で警告の承認をやり直してください")
    lines += [
        "",
        "  1. 証明書の警告を承認する（「詳細」→「アクセスする」系）",
        "  2. 言語を選んで「はじめる」→ カメラを許可",
        "  3. 端末を固定し、完了の音が鳴るまで前方を見る（基準の測定）",
        "",
        "  同じ Wi-Fi にいること。繋がらないときはファイアウォールで",
        f"  {port} の受信を許可してください。終了は Ctrl+C。",
        _RULE,
    ]
    return "\n".join(lines)


def _without_link_notice(config: dict[str, Any]) -> dict[str, Any]:
    """待ち受け側が URL を重ねて出さないようにした設定。"""
    source = dict(config.get("source", {}))
    remote = {**source.get("remote", {}), "announce": False}
    return {**config, "source": {**source, "remote": remote}}


if __name__ == "__main__":
    raise SystemExit(main())
