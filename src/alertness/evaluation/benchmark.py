"""録画 CSV で判定を測り直し、誤警告と見逃しを数える。

- 誤警告: 起きている人の収録を流し、警告（MEDIUM 以上）が出ていた時間の割合と回数。
  どの cue が警告を出していたかも数え、どこを直せば減るかを示す。
- 見逃し: 同じ収録に scenarios の状態を差し込み、上がるべき軸が上がったかと遅れ。
  差し込まない再生の同じ区間も数え、もともとの誤警告で「検出した」ことにしない。

rescore と同じく、CSV には特徴量が残っているので判定だけをやり直す。
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from .. import factory
from ..contracts import Assessment, CalibrationProfile, Features, Frame, Level, Observation
from ..features.ear import eye_openness
from ..rescore import _features_from
from ..temporal import TemporalContext
from .scenarios import Scenario

WARMUP_SECONDS = 60.0  # 基準が育つまでの区間。誤警告に数えない
INJECT_AT = 90.0  # 差し込みを始める時刻（収録の先頭から）
_ALERT = Level.MEDIUM


@dataclass(frozen=True)
class Sample:
    t: float  # 収録の先頭からの秒
    assessment: Assessment


def load_features(path: str, step: int = 1) -> list[Features]:
    """CSV の行を特徴量にする。古い収録に無い gaze_dx と eye_open は補う。"""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[::step]
    center = _gaze_center(rows)
    features = [_features_from(row) for row in rows]
    return [_backfill(f, center) for f in features if f is not None]


def _gaze_center(rows: Sequence[dict[str, str]]) -> float | None:
    # gaze_off = |gaze_x - 中心| なので、全フレームに共通して現れる候補が中心。
    votes: Counter[float] = Counter()
    for row in rows[:4000]:
        if row.get("gaze_x") and row.get("gaze_off"):
            x, off = float(row["gaze_x"]), float(row["gaze_off"])
            votes.update((round(x - off, 4), round(x + off, 4)))
    if not votes:
        return None
    # 視線が片側に偏り続けた収録では 2 候補が同票になりうる。登録順で決めず、
    # 正面の 0.5 に近い方を採る。
    top = max(votes.values())
    return min((c for c, n in votes.items() if n == top), key=lambda c: abs(c - 0.5))


def _backfill(f: Features, center: float | None) -> Features:
    values = dict(f.values)
    if center is not None and "gaze_dx" not in values and "gaze_x" in values:
        values["gaze_dx"] = values["gaze_x"] - center
    if "ear_norm" in values and "eye_open" not in values:
        values["eye_open"] = eye_openness(
            values["ear_norm"], values.get("eyeBlinkLeft"), values.get("eyeBlinkRight")
        )
    return Features(values=values, timestamp=f.timestamp, face_present=f.face_present)


def replay(
    frames: Sequence[Features], config: dict, scenario: Scenario | None = None
) -> Iterator[Sample]:
    """判定し直す。scenario があれば INJECT_AT から差し込む。"""
    classifier = factory.build_classifier(config)
    temporal = TemporalContext(max_seconds=60.0, fps=30.0)
    start = frames[0].timestamp if frames else 0.0
    for f in frames:
        t = f.timestamp - start
        inside = scenario is not None and INJECT_AT <= t < INJECT_AT + scenario.seconds
        if inside and f.face_present:
            changed = scenario.apply(t - INJECT_AT, f.values)
            f = _backfill(Features(changed, f.timestamp, True), None)
        temporal.append(f)
        obs = Observation(
            frame=Frame(image=None, index=0, timestamp=f.timestamp), landmarks=None,
            features=f, history=temporal, profile=CalibrationProfile.identity(),
        )
        yield Sample(t, classifier.assess(obs))


def false_alarms(samples: Sequence[Sample], config: dict) -> dict:
    """警告が出ていた時間の割合・1 時間あたりの回数・原因の cue の内訳。

    samples は差し込みなしの replay。起きている人の収録であることが前提で、
    警告はすべて誤警告として数える。
    """
    inverted = _inverted_dimensions(config)
    time: defaultdict[str, float] = defaultdict(float)
    startup: defaultdict[str, float] = defaultdict(float)
    causes: defaultdict[str, Counter[str]] = defaultdict(Counter)
    episodes: Counter[str] = Counter()
    last_alert: dict[str, float] = {}
    total, warmup, previous = 0.0, 0.0, None
    for sample in samples:
        dt = 0.0 if previous is None else min(0.5, sample.t - previous)
        previous = sample.t
        if sample.t < WARMUP_SECONDS:
            # 測り終えた直後は履歴が短く、窓の小さい判定が暴れやすい。別に数えて見落とさない。
            warmup += dt
            for name, dim in sample.assessment.dimensions.items():
                startup[name] += dt if dim.level >= _ALERT else 0.0
            continue
        total += dt
        for name, dim in sample.assessment.dimensions.items():
            if dim.level < _ALERT:
                continue
            if sample.t - last_alert.get(name, -1e9) > 3.0:
                episodes[name] += 1  # 3 秒以上空いたら別の 1 回
            last_alert[name] = sample.t
            time[name] += dt
            causes[name][_top_cue(sample.assessment, name, name in inverted)] += dt
    hours = total / 3600 if total > 0 else 1.0
    return {
        "seconds": total,
        "alert_share": {k: v / total for k, v in time.items()} if total > 0 else {},
        "startup_share": {k: v / warmup for k, v in startup.items()} if warmup > 0 else {},
        "episodes_per_hour": {k: v / hours for k, v in episodes.items()},
        "causes": {k: {c: s / total for c, s in v.items()} for k, v in causes.items()},
    }


def detection(
    frames: Sequence[Features], config: dict, scenario: Scenario, control: Sequence[Sample]
) -> dict:
    """差し込んだ区間で対象の軸が上がったか。control（差し込みなしの replay）と並べて返す。"""
    grace = 2.0
    injected = _window(replay(frames, config, scenario), scenario, grace)
    return {"injected": injected, "control": _window(iter(control), scenario, grace)}


def _window(samples: Iterator[Sample], scenario: Scenario, grace: float) -> dict:
    levels = [
        (s.t - INJECT_AT, s.assessment.dimensions[scenario.dimension].level)
        for s in samples
        if INJECT_AT - 3.0 <= s.t <= INJECT_AT + scenario.seconds + grace
        and scenario.dimension in s.assessment.dimensions
    ]
    before = [lv for t, lv in levels if t < 0]
    inside = [(t, lv) for t, lv in levels if t >= 0]
    hits = [t for t, lv in inside if lv >= _ALERT]
    return {
        "alerting_before": bool(before) and max(before) >= _ALERT,
        "latency": hits[0] if hits else None,
        "coverage": len(hits) / len(inside) if inside else 0.0,
    }


def _inverted_dimensions(config: dict) -> set[str]:
    dims = config.get("assessment", {}).get("dimensions", [])
    return {d["name"] for d in dims if d.get("alert_on") == "low"}


def _top_cue(assessment: Assessment, dimension: str, inverted: bool) -> str:
    cues = [c for c in assessment.cues if c.dimension == dimension]
    if not cues:
        return "-"
    strength: Callable[[float], float] = (lambda s: 1.0 - s) if inverted else (lambda s: s)
    return max(cues, key=lambda c: strength(c.score)).name
