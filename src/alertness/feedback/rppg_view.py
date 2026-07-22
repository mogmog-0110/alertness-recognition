"""左下の開発者向け rPPG 可視化（サーモグラフィ風）。

「額のどこが、どれだけ脈打っているか」を面で見せる。素朴に「現在フレーム − 時間平均」を出すと
センサノイズの方が脈より大きく、画面がただ目まぐるしく色を変えるだけになる。そこで:

- 速い平均と遅い平均の差を取り、心拍帯（およそ 0.25〜3Hz = 15〜200bpm）だけを通す。
- 空間方向にぼかす。1画素あたりの脈信号は極小なので、まとめて平均しないと埋もれる。
- 色の割り当てを自動調整しない。明るさに対する変化率(%)を固定スケールに載せるので、
  信号が無いときは中間色のまま静かになり、脈があるときだけ振れる。

判定には影響しない表示専用。
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
    """額の色を画素ごとに帯域通過し、明るさに対する変化率をサーモ風に描く（表示専用）。"""

    def __init__(
        self,
        fps: float = 30.0,
        alpha_fast: float = 0.5,
        alpha_slow: float = 0.05,
        span_pct: float = 0.5,
    ) -> None:
        # 2つの時間平均の差＝帯域通過。fps=30 でおよそ 0.25〜3.3Hz（15〜200bpm）が残る。
        self._alpha_fast = alpha_fast
        self._alpha_slow = alpha_slow
        self._span_pct = span_pct  # この変化率(%)で色が振り切れる。自動調整はしない。
        self._proc = (32, 20)  # 計算用の縮小サイズ。粗いほど1画素あたりの信号が強くなる。
        self._fast: np.ndarray | None = None
        self._slow: np.ndarray | None = None
        self._amp_pct = 0.0  # 直近の平均変化率(%)。拍動インジケータと数値表示に使う。

    def render(self, img: np.ndarray, obs: Observation) -> None:
        h, w = img.shape[:2]
        lm = obs.landmarks
        box = forehead_roi_box(lm, w, h) if lm.detected else None

        px, top = 16, h - 140
        _label(img, "rPPG view (forehead)", (px, top), 0.5, (255, 255, 255))
        cell_w, cell_h, gap = 120, 52, 8
        cy = top + 8

        if box is None:
            self._reset()
            _label(img, "no face", (px, cy + 28), 0.5, (0, 140, 255))
        else:
            patch = obs.frame.image[box[1] : box[3], box[0] : box[2]]
            if patch.size:
                heat = self._thermo(patch)
                raw = cv2.resize(patch, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                big = cv2.resize(heat, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
                _blit(img, raw, px, cy)
                _blit(img, big, px + cell_w + gap, cy)
                _label(img, "raw", (px + 4, cy + cell_h - 4), 0.4, (255, 255, 255))
                _label(img, "pulse", (px + cell_w + gap + 4, cy + cell_h - 4), 0.4, (255, 255, 255))
                self._draw_beat(img, px + 2 * cell_w + 2 * gap + 16, cy + cell_h // 2)

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
        _label(
            img,
            f"amp {self._amp_pct:+.3f}%  (scale +/-{self._span_pct:.1f}% of brightness)",
            (px, ty + 18),
            0.42,
            (200, 200, 200),
        )

    def _reset(self) -> None:
        self._fast = self._slow = None
        self._amp_pct = 0.0

    def _thermo(self, patch: np.ndarray) -> np.ndarray:
        # 縮小＋ぼかしで空間方向に均す → 速い平均と遅い平均の差＝心拍帯の成分。
        small = cv2.resize(patch, self._proc, interpolation=cv2.INTER_AREA).astype(np.float32)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        green = small[:, :, 1]  # 緑が血流の脈をよく表す

        fast, slow = self._fast, self._slow
        if fast is None or slow is None or fast.shape != green.shape:
            fast = slow = green.copy()
        band = fast - slow
        self._fast = fast + self._alpha_fast * (green - fast)
        self._slow = slow + self._alpha_slow * (green - slow)

        # 明るさに対する変化率(%)。露出や肌色に依らず、脈の強さを同じ尺度で見られる。
        pct = band / np.maximum(slow, 1.0) * 100.0
        self._amp_pct = float(np.mean(pct))
        norm = np.clip(128.0 + pct / self._span_pct * 127.0, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(norm, _CMAP)

    def _draw_beat(self, img: np.ndarray, x: int, y: int) -> None:
        # 額全体の平均変化率。拍動していれば、この丸が心拍のリズムで膨らんで色を変える。
        level = min(1.0, abs(self._amp_pct) / self._span_pct)
        shade = int(np.clip(128.0 + self._amp_pct / self._span_pct * 127.0, 0, 255))
        color = cv2.applyColorMap(np.full((1, 1), shade, np.uint8), _CMAP)[0, 0]
        cv2.circle(img, (x, y), 4 + int(10 * level), tuple(int(c) for c in color), -1)
        cv2.circle(img, (x, y), 14, (200, 200, 200), 1)


def _blit(img: np.ndarray, tile: np.ndarray, x: int, y: int) -> None:
    # 画像からはみ出す分はクリップする（小さい画面でも落ちないように）。
    height, width = img.shape[:2]
    th, tw = tile.shape[:2]
    x1, y1 = min(x + tw, width), min(y + th, height)
    if x >= width or y >= height or x1 <= x or y1 <= y:
        return
    img[y:y1, x:x1] = tile[: y1 - y, : x1 - x]
    cv2.rectangle(img, (x, y), (x1 - 1, y1 - 1), (0, 255, 255), 1)
