"""運転者に見せる画面。警告だけを大きく、それ以外は何も出さない。

実車では運転者はモニタを見ない（見てはいけない）。特徴量の数値もタイムラインも、
運転中に読めるものではないし、読もうとすること自体が脇見になる。デモでも同じ前提に
立ち、観察用の情報は別の画面（overlay.render）に置いて、こちらは警告に徹する。

夜間に明るい画面を正面に置くのはそれ自体が危険なので、警告が無い間は黒のまま、
動いていることを示す小さな点だけを出す。警告が出たときにだけ画面全体を使う。

文字は OpenCV の制約で日本語を描けないため英字表記（コメントとログは日本語）。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..contracts import Assessment, Level

WINDOW_NAME = "Driver"

# 段ごとの色（BGR）。注意喚起は橙、警告は赤。none/low は何も描かない。
_COLORS = {
    Level.MEDIUM: (0, 140, 255),
    Level.HIGH: (0, 0, 255),
}
_HEARTBEAT = (40, 40, 40)  # 生きていることを示す点。暗くして夜間の眩しさを避ける


def render(assessment: Assessment, size: tuple[int, int]) -> np.ndarray:
    """運転者向けの1枚を作る。size は (幅, 高さ)。

    警告が無ければ黒に点だけ。あれば枠と軸名を大きく描く。
    """
    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    dim = _worst(assessment)
    if dim is None:
        cv2.circle(canvas, (width - 20, height - 20), 4, _HEARTBEAT, -1)
        return canvas

    color = _COLORS[dim.level]
    thickness = max(8, round(min(width, height) * 0.06))
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), color, thickness)
    _centered(canvas, dim.display_name.upper(), 0.42, color, 0.0016)
    if dim.level >= Level.HIGH:
        # HIGH だけ言葉を足す。MEDIUM で毎回文章を出すと、読ませる時間を要求してしまう。
        _centered(canvas, "PULL OVER", 0.62, color, 0.0011)
    return canvas


def _worst(assessment: Assessment):
    """警告に値する軸のうち、最も段が高いもの。無ければ None。"""
    alerting = [d for d in assessment.dimensions.values() if d.level >= Level.MEDIUM]
    if not alerting:
        return None
    return max(alerting, key=lambda d: (d.level, d.alarm))


def _centered(img: np.ndarray, text: str, y_ratio: float, color: tuple, size_ratio: float) -> None:
    height, width = img.shape[:2]
    scale = max(0.6, width * size_ratio)
    thickness = max(2, round(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    org = ((width - tw) // 2, round(height * y_ratio) + th // 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
