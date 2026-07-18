"""左下の開発者向け rPPG 可視化。

「額のどこを、どんな色変化で見て、どこからストレスを出しているか」を目で追うための道具。
額ROIの切り出し（拡大）と、そこから取り出した脈波（＝ごく微小な色変化を増幅した波形）を出す。
状態（過去フレームの肌色バッファ）を持つので、描画関数群とは別にクラスにする。判定には一切
影響しない、純粋な表示。
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ..contracts import Observation
from ..features.rppg import forehead_roi_box, pos_signal

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img: np.ndarray, s: str, org: tuple[int, int], scale: float, color: tuple) -> None:
    cv2.putText(img, s, org, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, _FONT, scale, color, 1, cv2.LINE_AA)


def _fmt(name: str, value: float, unit: str) -> str:
    return f"{name} --{unit}" if np.isnan(value) else f"{name} {value:.0f}{unit}"


class RppgView:
    """額の肌色バッファを持ち、切り出し＋脈波を左下に描く（表示専用）。"""

    def __init__(self, fps: float = 30.0, window_seconds: float = 10.0) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=max(8, int(window_seconds * fps)))

    def render(self, img: np.ndarray, obs: Observation) -> None:
        h, w = img.shape[:2]
        lm = obs.landmarks
        box = forehead_roi_box(lm, w, h) if lm.detected else None
        if box is not None:
            self._buf.append(_roi_mean_rgb(obs.frame.image, box))

        px, pw = 16, 256
        crop_w, crop_h = 128, 40
        top = h - 120
        _label(img, "rPPG view (forehead)", (px, top), 0.5, (255, 255, 255))

        crop_y = top + 8
        if box is not None:
            _draw_crop(img, obs.frame.image, box, px, crop_y, crop_w, crop_h)
        else:
            _label(img, "no face", (px, crop_y + 26), 0.5, (0, 140, 255))

        tx = px + crop_w + 12
        hr = obs.features.get("hr_bpm", float("nan"))
        quality = obs.features.get("rppg_quality", float("nan"))
        hrv = obs.features.get("hrv_rmssd", float("nan"))
        _label(img, _fmt("HR", hr, "bpm"), (tx, crop_y + 14), 0.45, (0, 255, 255))
        _label(
            img,
            _fmt("Q", quality * 100 if not np.isnan(quality) else quality, "%"),
            (tx, crop_y + 32),
            0.45,
            (0, 200, 0),
        )
        _label(
            img,
            f"mode {'HRV' if not np.isnan(hrv) else 'HR'}",
            (tx, crop_y + 50),
            0.45,
            (255, 255, 255),
        )

        self._draw_waveform(img, px, top + 54, pw, 44)

    def _draw_waveform(self, img: np.ndarray, x: int, y: int, width: int, height: int) -> None:
        # 額の色から取り出した脈波。微小な色変化を全幅に正規化して見せる。
        cv2.rectangle(img, (x, y), (x + width, y + height), (40, 40, 40), -1)
        mid = y + height // 2
        cv2.line(img, (x, mid), (x + width, mid), (90, 90, 90), 1)
        if len(self._buf) < 4:
            return

        pulse = pos_signal(np.array(self._buf))
        tail = pulse[-width:]  # 1サンプル≒1px で右へ流れる
        peak = float(np.max(np.abs(tail)))
        if peak < 1e-8:
            return
        amp = (height // 2) - 2
        pts = [(x + i, int(mid - (v / peak) * amp)) for i, v in enumerate(tail[-width:])]
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, (0, 230, 120), 1, cv2.LINE_AA)

        # 直近の脈で明滅する増幅スウォッチ（微小変化を色の濃淡で体感する用）。
        level = int(128 + 127 * float(tail[-1]) / peak)
        cv2.rectangle(
            img,
            (x, y + height + 3),
            (x + width, y + height + 11),
            (0, max(0, min(255, level)), 0),
            -1,
        )


def _roi_mean_rgb(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    patch = image[y0:y1, x0:x1].reshape(-1, image.shape[2])[:, :3]
    return patch.mean(axis=0)[::-1].astype(float)  # BGR→RGB


def _draw_crop(
    img: np.ndarray,
    image: np.ndarray,
    box: tuple[int, int, int, int],
    x: int,
    y: int,
    out_w: int,
    out_h: int,
) -> None:
    x0, y0, x1, y1 = box
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return
    crop = cv2.resize(patch, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    img[y : y + out_h, x : x + out_w] = crop
    cv2.rectangle(img, (x, y), (x + out_w, y + out_h), (0, 255, 255), 1)
