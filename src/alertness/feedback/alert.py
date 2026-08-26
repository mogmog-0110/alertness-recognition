"""音による警告。評価軸ごとに別の音を鳴らし、段階に応じて鳴らし方を変える。

運転者は画面を見られないので、伝わるのは音だけになる。1種類の音を一定間隔で
鳴らすだけだと2つの失敗が起きる:
- MEDIUM も HIGH も同じに聞こえ、どれだけ切迫しているかが伝わらない。
- 無視され続けているのに間隔が変わらず、そのまま気づかれずに終わる。

そこで段階を音の形（notice=先頭の一音を小さく / warn=パターン全体）で分け、
HIGH が続く間は鳴らすたびに間隔を詰める。詰めっぱなしはうるさくて装置ごと
切られるので、min_interval_seconds で下限を置く。
"""

from __future__ import annotations

import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..contracts import Level
from .tone import make_chime_wav, shapes


@dataclass
class _Episode:
    """1回の警告のまとまり。段が NONE/LOW まで下がるまで続く。"""

    level: Level
    plays: int  # この警告で何回鳴らしたか。間隔を詰める根拠
    last_at: float


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
        self._cooldown = cooldown_seconds  # 注意喚起(MEDIUM)の間隔
        self._min_interval = min_interval_seconds  # どれだけ詰めてもこれより短くしない
        self._escalate = escalate_factor  # 警告(HIGH)が続く間、間隔にかける係数
        self._episodes: dict[str, _Episode] = {}
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
        if level < Level.MEDIUM:
            self._episodes.pop(name, None)  # 収まった。次は最初の1回から数え直す
            return

        now = time.monotonic()
        episode = self._episodes.get(name)
        if episode is None or level > episode.level:
            # 立った直後と、段が上がった直後は待たせない。
            self._play(name, level)
            self._episodes[name] = _Episode(level, 1, now)
            return

        episode.level = level
        if now - episode.last_at < self._interval(level, episode.plays):
            return
        self._play(name, level)
        episode.plays += 1
        episode.last_at = now

    def _interval(self, level: Level, plays: int) -> float:
        """次に鳴らすまでの間隔（秒）。"""
        if level < Level.HIGH:
            return self._cooldown  # 注意喚起は一定間隔で。急かす段ではない
        base = self._cooldown / 2.0
        return max(self._min_interval, base * self._escalate ** max(0, plays - 1))

    def _play(self, name: str, level: Level) -> None:
        shape = "warn" if level >= Level.HIGH else "notice"
        path = self._paths[name].get(shape) or next(iter(self._paths[name].values()))
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except (ImportError, RuntimeError):
            pass
