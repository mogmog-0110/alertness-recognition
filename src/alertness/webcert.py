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
from collections.abc import Sequence

# 検出したアドレスが端末から届かない構成（VPN、有線と Wi-Fi の併用）で、
# 設定ファイルを触らずに URL と証明書の IP を指定するための逃げ道。
HOST_ENV = "ALERTNESS_HOST"


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


def lan_addresses() -> list[str]:
    """この PC の IPv4 アドレス。既定経路のものを先頭に置く。

    既定経路の NIC が端末と同じネットワークとは限らない。候補を全部証明書に入れて
    案内にも並べておけば、外れたときに作り直しも再起動も要らない。
    """
    found: list[str] = []
    candidates = [local_ip()]
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        candidates += [str(info[4][0]) for info in infos]
    except OSError:
        pass
    for candidate in candidates:
        address = ipaddress.ip_address(candidate)
        if address.is_loopback or address.is_link_local or candidate in found:
            continue
        found.append(candidate)
    return found


def certificate_hosts(certfile: str) -> tuple[str, ...]:
    """証明書に入っている IP。読めなければ空。"""
    try:
        from cryptography import x509
        from cryptography.x509.oid import ExtensionOID

        with open(certfile, "rb") as handle:
            cert = x509.load_pem_x509_certificate(handle.read())
        san = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
        return tuple(str(a) for a in san.value.get_values_for_type(x509.IPAddress))
    except Exception:  # noqa: BLE001 - 読めない証明書は「合っていない」と同じ扱い
        return ()


def certificate_host(certfile: str) -> str:
    """証明書に入っている先頭の IP。読めなければ空文字。"""
    hosts = certificate_hosts(certfile)
    return hosts[0] if hosts else ""


def key_matches(certfile: str, keyfile: str) -> bool:
    """証明書と秘密鍵が対になっているか。

    書き込みの途中で落ちると、IP は合っているのに鍵だけ違う組が残る。IP だけを
    見ていると作り直されず、TLS の待ち受けが毎回失敗し続ける。
    """
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization

        with open(certfile, "rb") as handle:
            cert = x509.load_pem_x509_certificate(handle.read())
        with open(keyfile, "rb") as handle:
            key = serialization.load_pem_private_key(handle.read(), password=None)
        return cert.public_key().public_numbers() == key.public_key().public_numbers()  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - 読めない組は作り直す
        return False


def write_self_signed(
    hosts: str | Sequence[str], certfile: str, keyfile: str, days: int = 825
) -> None:
    """hosts を SAN に入れた自己署名証明書を書く。"""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    names = [hosts] if isinstance(hosts, str) else list(dict.fromkeys(hosts))
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, names[0])])
    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(days=1))
        .not_valid_after(now + _dt.timedelta(days=days))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(n)) for n in names]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    # 鍵を先に置き換える。証明書の置き換え前に落ちても古い証明書の IP が残るので、
    # 次の起動で ensure が作り直す。
    _replace(keyfile, key_pem)
    _replace(certfile, cert.public_bytes(serialization.Encoding.PEM))


def _replace(path: str, data: bytes) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "wb") as handle:
        handle.write(data)
    os.replace(temporary, path)


def advertised_host(host: str = "") -> str:
    """端末に案内する IP。明示 → 環境変数 → 既定経路の順に決める。"""
    return host or os.environ.get(HOST_ENV, "") or local_ip()


def ensure(certfile: str, keyfile: str, host: str = "") -> tuple[str, bool]:
    """証明書を今の IP に合わせる。

    @return (使う IP, 作り直したか)

    無ければ作り、IP が入っていなければ作り直す。DHCP で IP が変わると iOS が
    証明書を拒み、「ページは開けるのにカメラが出ない」という原因の見えない
    失敗になる。黙って直すのではなく、呼び出し側が作り直しを知らせられるように
    真偽値を返す。
    """
    host = advertised_host(host)
    if (
        os.path.isfile(certfile)
        and os.path.isfile(keyfile)
        and host in certificate_hosts(certfile)
        and key_matches(certfile, keyfile)
    ):
        return host, False
    write_self_signed([host, *lan_addresses()], certfile, keyfile)
    return host, True
