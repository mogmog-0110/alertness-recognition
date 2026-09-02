#!/usr/bin/env python3
"""ブラウザ版のための自己署名証明書を作る。

iOS Safari は HTTPS でないとカメラを許可しない。LAN 内で使うだけなので、
自己署名で足りる（端末側で初回に 1 回だけ警告を承認する）。

    python scripts/make_cert.py --host 192.168.1.10

出力した cert.pem / key.pem を config の source.iphone.certfile / keyfile に書く。
IP を証明書に入れるのは、iOS が「アドレスと一致しない証明書」を強く拒むため。
"""

from __future__ import annotations

import argparse
import os
import subprocess


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True, help="この PC の IP アドレス（ipconfig で確認）")
    ap.add_argument("--out", default="certs", help="出力先")
    ap.add_argument("--days", type=int, default=825, help="有効日数")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cert = os.path.join(args.out, "cert.pem")
    key = os.path.join(args.out, "key.pem")
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", key, "-out", cert, "-days", str(args.days),
            "-subj", f"/CN={args.host}",
            "-addext", f"subjectAltName=IP:{args.host}",
        ],
        check=True,
    )
    print(f"作成: {cert} / {key}")
    print("config の source.iphone に次を書いてください:")
    print(f"    certfile: {cert}")
    print(f"    keyfile: {key}")
    print("    web_root: web")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
