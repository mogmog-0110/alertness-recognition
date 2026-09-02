#!/usr/bin/env python3
"""ブラウザ版のための自己署名証明書を作る。

通常は起動時に自動で用意されるので、これを手で叩く必要はない。
IP を明示したいときや、作り直したいときに使う。

    python scripts/make_cert.py            # この PC の IP を自動で使う
    python scripts/make_cert.py --host 192.168.1.10
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from alertness.webcert import local_ip, write_self_signed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="", help="この PC の IP（省略時は自動検出）")
    ap.add_argument("--out", default="certs", help="出力先")
    ap.add_argument("--days", type=int, default=825, help="有効日数")
    args = ap.parse_args()

    host = args.host or local_ip()
    if host.startswith("127."):
        print("LAN アドレスが取れません。--host で明示してください。")
        return 1
    cert = os.path.join(args.out, "cert.pem")
    key = os.path.join(args.out, "key.pem")
    write_self_signed(host, cert, key, args.days)
    print(f"作成: {cert} / {key}  (IP {host})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
