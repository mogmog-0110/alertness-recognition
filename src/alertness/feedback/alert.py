"""音による警告。評価軸ごとに別の音を鳴らし、鳴らしすぎないよう間隔を空ける。"""

from __future__ import annotations

import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

from .tone import make_chime_wav


class AudioAlert:
    """評価軸名 → 音の対応を持ち、軸ごとに別の WAV を鳴らす。

    sounds は {"drowsiness": "drowsy", "distraction": "distracted"} のように
    軸名から音の種類への対応。鳴らす間隔(cooldown)は軸ごとに独立して数える。
    """

    def __init__(
        self,
        cooldown_seconds: float = 5.0,
        enabled: bool = True,
        sounds: Mapping[str, str] | None = None,
    ) -> None:
        self._enabled = enabled and sys.platform.startswith("win")
        self._cooldown = cooldown_seconds
        self._last: dict[str, float] = {}
        self._paths: dict[str, Path] = {}
        if self._enabled:
            self._prepare(sounds or {})

    def _prepare(self, sounds: Mapping[str, str]) -> None:
        tmp = Path(tempfile.gettempdir())
        for name, kind in sounds.items():
            try:
                path = tmp / f"alertness_{kind}.wav"
                if not path.exists():
                    make_chime_wav(path, kind)
                self._paths[name] = path
            except Exception:
                # 1つ作れなくても他は鳴らせるよう、その軸だけ諦める。
                continue

    def trigger(self, name: str) -> None:
        path = self._paths.get(name)
        if not self._enabled or path is None:
            return
        now = time.monotonic()
        if now - self._last.get(name, 0.0) < self._cooldown:
            return
        self._last[name] = now
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            pass
