"""デモ起動の案内。案内の後は何も出さない約束なので、案内そのものが正しいことを確かめる。"""

from __future__ import annotations

from alertness.demo import _banner, _without_link_notice


def test_the_banner_uses_the_configured_port():
    text = _banner("192.168.1.10", 9001, renewed=False, others=[])
    assert "https://192.168.1.10:9001/" in text
    assert "9001 の受信を許可" in text
    assert "8765" not in text


def test_other_addresses_and_a_renewed_certificate_are_mentioned():
    text = _banner("192.168.1.10", 8765, renewed=True, others=["10.0.0.5"])
    assert "https://10.0.0.5:8765/" in text
    assert "承認をやり直して" in text


def test_the_link_does_not_repeat_the_url():
    config = {"source": {"type": "remote", "remote": {"port": 8765}}, "feedback": {}}
    quiet = _without_link_notice(config)
    assert quiet["source"]["remote"] == {"port": 8765, "announce": False}
    assert "announce" not in config["source"]["remote"], "元の設定を書き換えない"
