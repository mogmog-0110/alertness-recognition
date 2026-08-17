"""映像入力を指定フレームレートの等間隔な時系列へダウンサンプリングする。

データセット固有の変換器やラベル形式には依存せず、動画から得た ``Frame`` の列を
CSV生成や評価へ渡す前に再標本化する。画像補間やフレーム複製は行わない。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import replace

from ..contracts import Frame


def validate_downsample_fps(source_fps: float, target_fps: float) -> None:
    """入力・目標FPSが有限の正数で、目標が入力以下であることを検証する。"""
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"入力FPSは有限の正数である必要があります: {source_fps!r}")
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError(f"CSV FPSは有限の正数である必要があります: {target_fps!r}")
    if target_fps > source_fps:
        raise ValueError(
            f"CSV FPS ({target_fps:g}) は入力動画FPS ({source_fps:g}) 以下である必要があります"
        )


def downsample_frames(
    frames: Iterable[Frame], source_fps: float, target_fps: float
) -> Iterator[Frame]:
    """目標時刻に到達した最初の入力フレームを、等間隔な時刻で返す。"""
    validate_downsample_fps(source_fps, target_fps)
    output_index = 0
    next_timestamp = 0.0
    epsilon = 1e-12
    for frame in frames:
        if frame.timestamp + epsilon < next_timestamp:
            continue
        yield replace(
            frame,
            index=output_index,
            timestamp=output_index / target_fps,
        )
        output_index += 1
        next_timestamp = output_index / target_fps


class DownsampledFrameSource:
    """``fps``と``frames()``を持つ入力源へ適用できる薄いアダプター。"""

    def __init__(self, source, target_fps: float) -> None:
        validate_downsample_fps(float(source.fps), float(target_fps))
        self._source = source
        self._source_fps = float(source.fps)
        self._fps = float(target_fps)

    @property
    def fps(self) -> float:
        return self._fps

    def frames(self) -> Iterator[Frame]:
        return downsample_frames(self._source.frames(), self._source_fps, self._fps)

    def close(self) -> None:
        self._source.close()
