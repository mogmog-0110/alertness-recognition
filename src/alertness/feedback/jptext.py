"""映像に日本語を描く。

OpenCV の putText は日本語を描けないので、ガイド指示など日本語が必要な箇所だけ
Pillow とシステムフォントで描く。Pillow が無い場合は何も描かない（呼び出し側が
ASCII の補助表示を併用しているので最低限の情報は残る）。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

# Windows に標準で入っている日本語フォント候補。
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
)


@lru_cache(maxsize=8)
def _load_font(size: int):
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def put_japanese(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    size: int = 26,
    color: tuple = (255, 255, 255),
    center: bool = False,
) -> None:
    # color は BGR で受け取り（cv2 と揃える）、Pillow 用に RGB へ変換する。
    # center=True なら各行を画面幅で水平センタリングする（org の x は無視）。
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return  # Pillow が無ければ描かない

    pil = Image.fromarray(img[:, :, ::-1])
    draw = ImageDraw.Draw(pil)
    font = _load_font(size)
    rgb = (color[2], color[1], color[0])
    x, y = org
    for i, line in enumerate(text.split("\n")):
        line_x = (img.shape[1] - int(draw.textlength(line, font=font))) // 2 if center else x
        line_y = y + i * (size + 6)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((line_x + dx, line_y + dy), line, font=font, fill=(0, 0, 0))
        draw.text((line_x, line_y), line, font=font, fill=rgb)
    img[:, :] = np.asarray(pil)[:, :, ::-1]
