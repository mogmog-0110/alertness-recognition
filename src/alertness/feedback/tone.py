"""警告音をその場で生成する。

評価軸ごとに音を変えるため、kind を指定して別パターンの WAV を作る。
- drowsy（眠気）: 下降する3音で強めに鳴らす。
- distracted（注意散漫）: 軽い高音2音でそっと知らせる。
クリック音を防ぐため前後にフェードをかける。生成は起動時に一度だけ。
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

# kind ごとの (周波数Hz, 長さ秒) の並び。
_PATTERNS = {
    "drowsy": ((784.0, 0.14), (588.0, 0.14), (440.0, 0.22)),
    "distracted": ((1047.0, 0.10), (1318.0, 0.13)),
    "ready": ((523.0, 0.12),),  # ガイド: 次の準備（一音）
    "go": ((659.0, 0.10), (988.0, 0.14)),  # ガイド: 開始（上がる二音）
}
_SAMPLE_RATE = 44100
_AMPLITUDE = 0.25  # 控えめにする


def make_chime_wav(path: str | Path, kind: str = "drowsy") -> None:
    notes = _PATTERNS.get(kind, _PATTERNS["drowsy"])
    segments = []
    for freq, duration in notes:
        n = int(_SAMPLE_RATE * duration)
        t = np.linspace(0.0, duration, n, endpoint=False)
        segments.append(_AMPLITUDE * _envelope(n) * np.sin(2.0 * np.pi * freq * t))

    audio = np.concatenate(segments)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def _envelope(n: int) -> np.ndarray:
    # 立ち上がり10ms・減衰40msのフェード。
    env = np.ones(n)
    fade_in = min(n, int(0.01 * _SAMPLE_RATE))
    fade_out = min(n, int(0.04 * _SAMPLE_RATE))
    env[:fade_in] = np.linspace(0.0, 1.0, fade_in)
    env[-fade_out:] = np.linspace(1.0, 0.0, fade_out)
    return env
