"""証明書とアドレス検出のテスト。

IP は DHCP で変わる。証明書の IP が現在と違うと iOS は接続を拒み、
「ページは開けるのにカメラが出ない」という原因の見えない失敗になる。
"""

from __future__ import annotations

import ipaddress

from alertness.webcert import certificate_host, ensure, local_ip, write_self_signed


def test_the_local_address_is_a_real_ipv4():
    address = local_ip()
    ipaddress.ip_address(address)  # 形式が壊れていれば例外


def test_the_certificate_carries_the_address(tmp_path):
    # iOS はアドレスと一致しない証明書を強く拒むので、SAN に IP が要る。
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    write_self_signed("192.168.1.10", str(cert), str(key))
    assert certificate_host(str(cert)) == "192.168.1.10"


def test_ensure_creates_what_is_missing(tmp_path):
    cert = tmp_path / "certs" / "cert.pem"
    key = tmp_path / "certs" / "key.pem"
    host, renewed = ensure(str(cert), str(key), "192.168.1.10")
    assert (host, renewed) == ("192.168.1.10", True)
    assert cert.is_file() and key.is_file()


def test_ensure_keeps_a_matching_certificate(tmp_path):
    # 合っているなら作り直さない。作り直すと端末の承認がやり直しになる。
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    ensure(str(cert), str(key), "192.168.1.10")
    before = cert.read_bytes()
    host, renewed = ensure(str(cert), str(key), "192.168.1.10")
    assert (host, renewed) == ("192.168.1.10", False)
    assert cert.read_bytes() == before


def test_ensure_replaces_a_stale_certificate(tmp_path):
    # IP が変わったら作り直す。放置すると iOS が拒み、原因が見えない。
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    ensure(str(cert), str(key), "192.168.1.10")
    host, renewed = ensure(str(cert), str(key), "192.168.1.99")
    assert (host, renewed) == ("192.168.1.99", True)
    assert certificate_host(str(cert)) == "192.168.1.99"
