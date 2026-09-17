"""音による警告。評価軸ごとに別の音を鳴らし、段階に応じて鳴らし方を変える。

運転者は画面を見られないので、伝わるのは音だけになる。段階は音の形
（notice=先頭の一音を小さく / warn=パターン全体）で分ける。いつ鳴らすかは
AlertCadence が決め、ここは PC のスピーカーで鳴らす部分だけを持つ。
"""

from __future__ import annotations

import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

from ..contracts import Level
from .cadence import AlertCadence
from .tone import make_chime_wav, shapes


class AudioAlert:
    """評価軸名 → 音の対応を持ち、軸ごとに別の WAV を鳴らす。

    sounds は {"drowsiness": "drowsy", "distraction": "distracted"} のように
    軸名から音の種類への対応。鳴らす間隔は軸ごとに独立して数える。
    """

    def __init__(
        self,
        cooldown_seconds: float = 5.0,
        enabled: bool = True,
        sounds: Mapping[str, str] | None = None,
        min_interval_seconds: float = 1.5,
        escalate_factor: float = 0.7,
    ) -> None:
        self._enabled = enabled and sys.platform.startswith("win")
        self._cadence = AlertCadence(cooldown_seconds, min_interval_seconds, escalate_factor)
        self._paths: dict[str, dict[str, Path]] = {}
        if self._enabled:
            self._prepare(sounds or {})

    def _prepare(self, sounds: Mapping[str, str]) -> None:
        tmp = Path(tempfile.gettempdir())
        for name, kind in sounds.items():
            built: dict[str, Path] = {}
            for shape in shapes():
                try:
                    path = tmp / f"alertness_{kind}_{shape}.wav"
                    if not path.exists():
                        make_chime_wav(path, kind, shape)
                    built[shape] = path
                except OSError:
                    # 1つ作れなくても他は鳴らせるよう、その段だけ諦める。
                    continue
            if built:
                self._paths[name] = built

    def trigger(self, name: str, level: Level) -> None:
        """その軸の現在の段を伝える。鳴らすかどうかはここで決める。

        段に関わらず毎フレーム呼ぶこと。収まったことも伝わらないと、次に立ったときに
        「続きの警告」と誤解して詰めた間隔から鳴り始める。
        """
        if not self._enabled or name not in self._paths:
            return
        if self._cadence.due(name, level, time.monotonic()):
            self._play(name, level)

    def _play(self, name: str, level: Level) -> None:
        shape = "warn" if level >= Level.HIGH else "notice"
        path = self._paths[name].get(shape) or next(iter(self._paths[name].values()))
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except (ImportError, RuntimeError):
            pass
