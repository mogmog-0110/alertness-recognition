"""iPhone の代わりに映像を送る確認用のクライアント。

端末が無くても Python 側だけで経路を確かめられるようにする。片方の窓で
`type: iphone` にしたアプリを動かし、もう片方でこれを実行すると、フレームを送って
返ってきた判定を表示する。ファイアウォールや配線の切り分けにも使える。

    python examples/iphone_fake_device.py --url ws://127.0.0.1:8765 --video sample.mp4

--video を省くと合成画像を送る（顔は写っていないので判定は「見失い」側に倒れる）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import time

import cv2
import numpy as np

_HEADER = struct.Struct("<d")  # 撮影時刻（秒, float64 リトルエンディアン）


def _frames(path: str | None, count: int, size: tuple[int, int]):
    """送る画像を順に返す。動画が尽きたら先頭へ戻る。"""
    if not path:
        blank = np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        for _ in range(count):
            yield blank
        return
    capture = cv2.VideoCapture(path)
    try:
        for _ in range(count):
            ok, image = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"動画を読めませんでした: {path}")
            yield image
    finally:
        capture.release()


async def _run(args: argparse.Namespace) -> None:
    from websockets.asyncio.client import connect

    async with connect(args.url) as ws:
        print(f"[接続] {args.url}")

        async def show() -> None:
            async for message in ws:
                reply = json.loads(message)
                mark = "★" if reply.get("alert") else " "
                print(f"{mark} {reply.get('level'):>6} {reply.get('message', '')}")

        reader = asyncio.create_task(show())
        started = time.perf_counter()
        for index, image in enumerate(_frames(args.video, args.count, (640, 480))):
            ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not ok:
                raise RuntimeError("JPEG に変換できませんでした。")
            # 撮影時刻は送り手が入れる。受信時刻で代用すると、通信の揺らぎが
            # そのまま生理指標の歪みになる。
            await ws.send(_HEADER.pack(index / args.fps) + bytes(buffer))
            await asyncio.sleep(max(0.0, (index + 1) / args.fps - (time.perf_counter() - started)))
        await asyncio.sleep(0.5)  # 最後の判定が返るのを待つ
        reader.cancel()


def main() -> int:
    parser = argparse.ArgumentParser(description="iPhone の代わりに映像を送る確認用クライアント")
    parser.add_argument("--url", default="ws://127.0.0.1:8765", help="接続先")
    parser.add_argument("--video", default=None, help="送る動画。省くと合成画像を送る")
    parser.add_argument("--fps", type=float, default=15.0, help="送る速さ（既定: 15）")
    parser.add_argument("--count", type=int, default=150, help="送る枚数")
    parser.add_argument("--quality", type=int, default=80, help="JPEG の品質")
    asyncio.run(_run(parser.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
