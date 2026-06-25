"""設定の読み込み。YAML を辞書として読むだけの薄い層。

しきい値などの中身は config/default.yaml が持つ。ここでは存在と形式だけ確認する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {p}")
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("設定ファイルの形式が不正です（先頭はマッピングである必要があります）。")
    return data
