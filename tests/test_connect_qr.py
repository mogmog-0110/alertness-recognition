"""接続用 QR のテスト。

描けたかではなく、読み取って元の URL に戻るかを確かめる。OpenCV の読み取りは
周囲の余白が無くても読めてしまうので、余白の削りすぎはここでは捕まらない。
"""

from __future__ import annotations

import cv2

from alertness.feedback.connect_qr import render


def _decode_all(image) -> set[str]:
    ok, texts, _, _ = cv2.QRCodeDetector().detectAndDecodeMulti(image)
    return set(texts) if ok else set()


def test_the_code_reads_back_as_the_url():
    url = "https://172.21.9.188:8765/"
    assert _decode_all(render(url)) == {url}
