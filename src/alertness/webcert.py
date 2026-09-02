"""ブラウザ版のための自己署名証明書と、この PC の LAN アドレス。

ブラウザは HTTPS でないとカメラを許可しない。LAN 内で使うだけなので自己署名で
足りるが、**IP を証明書に入れる必要がある**（iOS はアドレスと一致しない証明書を
強く拒む）。IP は DHCP で変わるので、手で調べて手で作り直す運用は続かない。

ここでアドレスの検出と作り直しをまとめて面倒を見る。openssl コマンドには頼らない
（Windows には入っていないことが多い）。
"""

from __future__ import annotations

import datetime as _dt
import ipaddress
import os
import socket


def local_ip() -> str:
    """この PC の LAN アドレス。

    外へ UDP ソケットを「繋ぐ」だけで実際には送らない。経路表を引く目的なので
    相手に到達する必要はなく、オフラインでも既定の経路があれば取れる。
    hostname からの逆引きは 127.0.0.1 を返すことがあるので使わない。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 9))  # TEST-NET-1。実在しない前提のアドレス
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def certificate_host(certfile: str) -> str:
    """証明書に入っている IP。読めなければ空文字。"""
    try:
        from cryptography import x509
        from cryptography.x509.oid import ExtensionOID

        with open(certfile, "rb") as handle:
            cert = x509.load_pem_x509_certificate(handle.read())
        san = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
        addresses = san.value.get_values_for_type(x509.IPAddress)
        return str(addresses[0]) if addresses else ""
    except Exception:  # noqa: BLE001 - 読めない証明書は「合っていない」と同じ扱い
        return ""


def write_self_signed(host: str, certfile: str, keyfile: str, days: int = 825) -> None:
    """host を SAN に入れた自己署名証明書を書く。"""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)])
    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(days=1))
        .not_valid_after(now + _dt.timedelta(days=days))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(host))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    for path in (certfile, keyfile):
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
    with open(certfile, "wb") as handle:
        handle.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(keyfile, "wb") as handle:
        handle.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )


def ensure(certfile: str, keyfile: str, host: str = "") -> tuple[str, bool]:
    """証明書を今の IP に合わせる。

    @return (使う IP, 作り直したか)

    無ければ作り、IP が変わっていれば作り直す。DHCP で IP が変わると iOS が
    証明書を拒み、「ページは開けるのにカメラが出ない」という原因の見えない
    失敗になる。黙って直すのではなく、呼び出し側が作り直しを知らせられるように
    真偽値を返す。
    """
    host = host or local_ip()
    if os.path.isfile(certfile) and os.path.isfile(keyfile):
        if certificate_host(certfile) == host:
            return host, False
    write_self_signed(host, certfile, keyfile)
    return host, True
