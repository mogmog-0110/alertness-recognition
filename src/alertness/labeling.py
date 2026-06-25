"""録画ラベルの実行時状態と、数字キー→ラベルの対応。

ポーズ中は画面を見られないので、手元のキーで「いまの状態」を指定して
そのフレームに正解ラベルを付けられるようにする。ラベル無し("")の区間は
採点対象から外れる。
"""

from __future__ import annotations

from collections.abc import Sequence


class LabelState:
    """いま記録するラベルを保持する可変の入れ物。アプリが書き換え、録画が読む。"""

    def __init__(self, value: str = "") -> None:
        self.value = value


def key_label_map(dimension_names: Sequence[str], awake: str = "awake") -> dict[int, str]:
    # 0=ラベル無し, 1=awake, 2以降=各評価軸（最大9まで）。
    mapping = {ord("0"): "", ord("1"): awake}
    for offset, name in enumerate(dimension_names, start=2):
        if offset > 9:
            break
        mapping[ord(str(offset))] = name
    return mapping
