"""端末に出す進行ログの出し分け。

デモの操作者が見るのは起動時の案内だけでよい。接続の出入りや命令の受信は
繋がらないときの切り分けにしか使わないので、--verbose を付けたときだけ出す。
人が手を打つべき異常は、この仕組みを通さず常に出す。
"""

from __future__ import annotations

_verbose = False


def set_verbose(enabled: bool) -> None:
    global _verbose
    _verbose = bool(enabled)


def is_verbose() -> bool:
    return _verbose


def detail(message: str) -> None:
    if _verbose:
        print(message, flush=True)
