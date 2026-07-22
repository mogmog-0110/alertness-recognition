"""左下の開発者向け rPPG 可視化（サーモグラフィ風）。

「額のどこが、どれだけ脈打っているか」を面で見せる。各画素の色を時間平均（EMA）で追い、
その差分＝速い成分（脈）を増幅してカラーマップに載せる（簡易オイラー動画増幅）。肉眼では
見えない微小な色変化が、額の上を波打つ熱分布のように可視化される。判定には影響しない表示専用。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..contracts import Observation
from ..features.rppg import forehead_roi_box

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_CMAP = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)


def _label(img: np.ndarray, s: str, org: tuple[int, int], scale: float, color: tuple) -> None:
    cv2.putText(img, s, org, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, _FONT, scale, color, 1, cv2.LINE_AA)


def _fmt(name: str, value: float, unit: str) -> str:
    return f"{name} --{unit}" if np.isnan(value) else f"{name} {value:.0f}{unit}"


class RppgView:
    """額の色を画素ごとに時間平均で追い、脈成分を増幅してサーモ風に描く（表示専用）。"""

    def __init__(self, fps: float = 30.0, alpha: float = 0.15, gain_sigma: float = 55.0) -> None:
        self._alpha = alpha  # 時間平均(EMA)の追従。大きいほど速い変化まで平均に取り込む。
        self._gain_sigma = gain_sigma  # 1σ の色変化をこの幅で色に割り当てる（増幅率）。
        self._proc = (64, 40)  # 計算用の縮小サイズ
        self._mean: np.ndarray | None = None  # 画素ごとの色の時間平均

    def render(self, img: np.ndarray, obs: Observation) -> None:
        h, w = img.shape[:2]
        lm = obs.landmarks
        box = forehead_roi_box(lm, w, h) if lm.detected else None

        px, top = 16, h - 122
        _label(img, "rPPG view (forehead)", (px, top), 0.5, (255, 255, 255))
        cell_w, cell_h, gap = 120, 52, 8
        cy = top + 8

        if box is None:
            self._mean = None  # 顔を見失ったら平均をやり直す。
            _label(img, "no face", (px, cy + 28), 0.5, (0, 140, 255))
        else:
            patch = obs.frame.image[box[1] : box[3], box[0] : box[2]]
            if patch.size:
                heat = self._thermo(patch)
                _blit(
                    img,
                    cv2.resize(patch, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST),
                    px,
                    cy,
                )
                _blit(
                    img,
                    cv2.resize(heat, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST),
                    px + cell_w + gap,
                    cy,
                )
                _label(img, "raw", (px + 4, cy + cell_h - 4), 0.4, (255, 255, 255))
                _label(img, "pulse", (px + cell_w + gap + 4, cy + cell_h - 4), 0.4, (255, 255, 255))

        ty = cy + cell_h + 16
        hr = obs.features.get("hr_bpm", float("nan"))
        quality = obs.features.get("rppg_quality", float("nan"))
        hrv = obs.features.get("hrv_rmssd", float("nan"))
        q_pct = quality * 100 if not np.isnan(quality) else quality
        mode = "HRV" if not np.isnan(hrv) else "HR"
        _label(
            img,
            f"{_fmt('HR', hr, 'bpm')}  {_fmt('Q', q_pct, '%')}  mode {mode}",
            (px, ty),
            0.45,
            (0, 255, 255),
        )

    def _thermo(self, patch: np.ndarray) -> np.ndarray:
        # 縮小 → 画素ごとの時間平均との差（脈成分）→ σで正規化 → カラーマップ。
        small = cv2.resize(patch, self._proc, interpolation=cv2.INTER_AREA).astype(np.float32)
        mean = self._mean
        if mean is None or mean.shape != small.shape:
            mean = small.copy()
        dev = small[:, :, 1] - mean[:, :, 1]  # 緑が血流の脈をよく表す
        self._mean = (1.0 - self._alpha) * mean + self._alpha * small

        std = float(np.std(dev))
        if std < 1e-6:
            norm = np.full(dev.shape, 128, dtype=np.uint8)
        else:
            norm = np.clip(128.0 + (dev / std) * self._gain_sigma, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(norm, _CMAP)


def _blit(img: np.ndarray, tile: np.ndarray, x: int, y: int) -> None:
    # 画像からはみ出す分はクリップする（小さい画面でも落ちないように）。
    height, width = img.shape[:2]
    th, tw = tile.shape[:2]
    x1, y1 = min(x + tw, width), min(y + th, height)
    if x >= width or y >= height or x1 <= x or y1 <= y:
        return
    img[y:y1, x:x1] = tile[: y1 - y, : x1 - x]
    cv2.rectangle(img, (x, y), (x1 - 1, y1 - 1), (0, 255, 255), 1)
