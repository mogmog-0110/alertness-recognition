"""実機で録った収録に「この区間はこう判定されるべき」を書き、判定を変えるたびに確かめる。

スマホで一度確かめた動き（目を閉じた後に警告が消える、横を向いて顔が外れたら脇見、など）は、
判定の中身を変えると黙って壊れうる。カメラの読み取りや通信を変えない限り、録った特徴量を
流し直せば実機と同じ判定になるので、毎回スマホで試し直さなくても済む。

期待の書き方（JSON）:
    {"recordings": [{"csv": "session_x.csv", "expect": [
        {"kind": "alert", "dimension": "drowsiness", "from": 4.5, "to": 6.0, "note": "..."},
        {"kind": "quiet", "dimension": "drowsiness", "from": 35.5, "to": 41.5, "note": "..."},
        {"kind": "headline", "dimension": "distraction", "at": 44.0, "note": "..."}]}]}
時刻は収録の先頭からの秒。alert は区間内のどこかで MEDIUM 以上、quiet は区間中ずっと
MEDIUM 未満、headline はその時刻に警告が出ていて、端末に出る軸がそれであること。
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

from ..contracts import Level
from .benchmark import Sample, load_features, replay


@dataclass(frozen=True)
class Outcome:
    csv: str
    expectation: dict
    passed: bool
    observed: str  # 外れたときに何が起きていたか


def load(path: str) -> list[tuple[str, list[dict]]]:
    """(CSV の絶対パス, 期待の並び) の並び。CSV は JSON からの相対パスで書く。"""
    with open(path, encoding="utf-8") as handle:
        spec = json.load(handle)
    base = os.path.dirname(os.path.abspath(path))
    return [(os.path.join(base, r["csv"]), r["expect"]) for r in spec["recordings"]]


def check_recording(csv_path: str, expectations: Sequence[dict], config: dict) -> list[Outcome]:
    samples = list(replay(load_features(csv_path), config))
    name = os.path.basename(csv_path)
    return [Outcome(name, e, *_check(samples, e)) for e in expectations]


def _check(samples: Sequence[Sample], expectation: dict) -> tuple[bool, str]:
    kind, dimension = expectation["kind"], expectation["dimension"]
    if kind == "headline":
        at = next((s for s in samples if s.t >= expectation["at"]), None)
        if at is None:
            return False, "収録がその時刻まで無い"
        head = at.assessment.headline()
        raised = head is not None and at.assessment.alert_level() >= Level.MEDIUM
        shown = head.name if head is not None and raised else "-"
        return shown == dimension, f"{at.t:.1f} 秒に出ていた軸: {shown}"
    inside = [s for s in samples if expectation["from"] <= s.t <= expectation["to"]]
    if not inside:
        return False, "収録がその区間まで無い"
    alerting = [s.t for s in inside if s.assessment.dimensions[dimension].level >= Level.MEDIUM]
    if kind == "alert":
        return bool(alerting), "一度も MEDIUM 以上にならなかった" if not alerting else ""
    if kind == "quiet":
        return not alerting, f"{alerting[0]:.1f}〜{alerting[-1]:.1f} 秒に警告" if alerting else ""
    raise ValueError(f"未知の kind: {kind}")
