"""CSVパスの解決。

Windows の cmd はワイルドカードを展開しないので、`runs\\session_*.csv` のような
指定を自前で展開する。引数省略時は runs/ を対象にし、素のファイル名は runs/ も探す。
"""

from __future__ import annotations

import glob
from collections.abc import Sequence
from pathlib import Path


def resolve_csv_paths(args: Sequence[str], default_dir: str = "runs") -> list[str]:
    raw = list(args) if args else [default_dir]
    found: list[str] = []
    for item in raw:
        path = Path(item)
        if path.is_dir():
            found.extend(sorted(str(p) for p in path.glob("*.csv")))
        elif any(c in item for c in "*?["):
            found.extend(sorted(glob.glob(item)))
        elif path.exists():
            found.append(str(path))
        else:
            alt = Path(default_dir) / path.name  # 素のファイル名なら runs/ も探す
            if alt.exists():
                found.append(str(alt))

    seen: set[str] = set()
    unique: list[str] = []
    for p in found:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique
