"""証明書とアドレス検出のテスト。

IP は DHCP で変わる。証明書の IP が現在と違うと iOS は接続を拒み、
「ページは開けるのにカメラが出ない」という原因の見えない失敗になる。
"""

from __future__ import annotations

import ipaddress

from alertness.webcert import (
    HOST_ENV,
    advertised_host,
    certificate_host,
    certificate_hosts,
    ensure,
    key_matches,
    lan_addresses,
    local_ip,
    write_self_signed,
)


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


def test_every_candidate_address_is_in_one_certificate(tmp_path):
    # 既定経路の NIC が端末と同じネットワークとは限らない。候補を全部入れておけば、
    # 案内の別の URL で開き直しても承認のやり直しが要らない。
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    write_self_signed(["192.168.1.10", "10.0.0.5"], str(cert), str(key))
    assert certificate_hosts(str(cert)) == ("192.168.1.10", "10.0.0.5")
    assert ensure(str(cert), str(key), "10.0.0.5") == ("10.0.0.5", False)


def test_a_key_that_does_not_match_is_replaced(tmp_path):
    # 書き込みの途中で落ちると IP は合っているのに鍵だけ違う組が残り、TLS が毎回失敗する。
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    other = tmp_path / "other"
    other.mkdir()
    write_self_signed("192.168.1.10", str(cert), str(key))
    write_self_signed("192.168.1.10", str(other / "cert.pem"), str(other / "key.pem"))
    key.write_bytes((other / "key.pem").read_bytes())
    assert not key_matches(str(cert), str(key))
    assert ensure(str(cert), str(key), "192.168.1.10") == ("192.168.1.10", True)
    assert key_matches(str(cert), str(key))


def test_the_advertised_host_can_be_set_from_the_environment(monkeypatch):
    monkeypatch.setenv(HOST_ENV, "10.1.2.3")
    assert advertised_host() == "10.1.2.3"
    assert advertised_host("192.168.1.10") == "192.168.1.10"


def test_lan_addresses_start_with_the_default_route():
    addresses = lan_addresses()
    if local_ip().startswith("127."):
        return  # オフラインの CI では既定経路が無い
    assert addresses[0] == local_ip()
    assert all(not a.startswith("127.") for a in addresses)
