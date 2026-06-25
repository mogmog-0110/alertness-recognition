"""ガイド付き収録の合図音。区切りごとに鳴らして、画面を見なくても切替が分かるように。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from .tone import make_chime_wav

_KINDS = ("ready", "go")


class CuePlayer:
    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled and sys.platform.startswith("win")
        self._paths: dict[str, Path] = {}
        if self._enabled:
            tmp = Path(tempfile.gettempdir())
            for kind in _KINDS:
                try:
                    path = tmp / f"alertness_{kind}.wav"
                    if not path.exists():
                        make_chime_wav(path, kind)
                    self._paths[kind] = path
                except Exception:
                    continue

    def play(self, kind: str) -> None:
        path = self._paths.get(kind)
        if not self._enabled or path is None:
            return
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            pass
