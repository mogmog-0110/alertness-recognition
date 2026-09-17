"""起きている人の特徴量に「危険な状態」を差し込む台本。

眠気や脇見の正解付き収録が手元に無くても、見逃しと遅れを測れるようにする。差し込む先は
実際の収録なので、瞬きのばらつき・姿勢の揺れ・検出の揺らぎは本物のまま残る。台本の形は
居眠り・脇見の研究でよく使われる典型に寄せてある（数値は各関数の中）。

どれも「差し込み始めてからの経過秒」と特徴量の袋を受け取り、書き換えた新しい袋を返す。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

Values = Mapping[str, float]


@dataclass(frozen=True)
class Scenario:
    name: str
    dimension: str  # 上がるべき軸。上がってはいけない台本は expect_alert=False
    seconds: float
    apply: Callable[[float, Values], dict[str, float]]
    expect_alert: bool = True


def _ramp(t: float, start: float, end: float) -> float:
    if t <= start:
        return 0.0
    return 1.0 if t >= end else (t - start) / (end - start)


def _blink(t: float, close: float, hold: float, reopen: float, depth: float) -> float | None:
    """1 回の瞬きの開き具合（1.0 → depth → 1.0）。瞬きの外なら None。"""
    if t < 0 or t > close + hold + reopen:
        return None
    if t < close:
        return 1.0 - (1.0 - depth) * t / close
    if t < close + hold:
        return depth
    return depth + (1.0 - depth) * (t - close - hold) / reopen


def _eyes(values: Values, openness: float) -> dict[str, float]:
    # EAR と瞬きスコアを同じ向きに動かす。片方だけ動かすと eye_open は閉じない。
    blink = max(0.0, min(1.0, (1.0 - openness) / 0.75))
    out = {**values, "ear_norm": openness, "eyeBlinkLeft": blink, "eyeBlinkRight": blink}
    return {k: v for k, v in out.items() if k != "eye_open"}


def _add(values: Values, key: str, delta: float) -> dict[str, float]:
    return {**values, key: values.get(key, 0.0) + delta}


def _slow_blinks(t: float, v: Values) -> dict[str, float]:
    openness = _blink(t % 4.0, 0.12, 0.5, 0.5, 0.25)
    return _eyes(v, openness) if openness is not None else dict(v)


def _drowsy_nods(t: float, v: Values) -> dict[str, float]:
    # まぶたが下がってから頭が 15 度落ち、はっと戻す。5 秒ごと。
    phase = t % 5.0
    nodded = _add(v, "pitch_rel", 15.0 * (_ramp(phase, 0.3, 0.8) - _ramp(phase, 0.8, 1.6)))
    openness = _blink(phase, 0.2, 0.6, 0.3, 0.35)
    return _eyes(nodded, openness) if openness is not None else nodded


def _awake_nods(t: float, v: Values) -> dict[str, float]:
    # 目を開けたままの相づち。3 秒ごとに 12 度。
    phase = t % 3.0
    return _add(v, "pitch_rel", 12.0 * (_ramp(phase, 0, 0.3) - _ramp(phase, 0.3, 0.7)))


def _gaze_away(t: float, v: Values) -> dict[str, float]:
    dx = v.get("gaze_dx", 0.0) + 0.12 * _ramp(t, 0, 0.15)
    return {**v, "gaze_dx": dx, "gaze_off": abs(dx)}


def _glances(t: float, v: Values) -> dict[str, float]:
    # 2 秒ごとに 1.2 秒よそ見する視覚的時分割。1 回ずつは短いが残高が戻りきらない。
    return _add(v, "yaw_rel", 35.0 if (t % 2.0) < 1.2 else 0.0)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("microsleep", "drowsiness", 2.0, lambda t, v: _eyes(v, 0.25)),
    Scenario("slow_blinks", "drowsiness", 60.0, _slow_blinks),
    Scenario("half_closed", "drowsiness", 40.0,
             lambda t, v: _eyes(v, v.get("ear_norm", 1.0) * 0.5)),
    Scenario("drowsy_nods", "drowsiness", 30.0, _drowsy_nods),
    Scenario("awake_nods", "drowsiness", 30.0, _awake_nods, expect_alert=False),
    Scenario("dozing_head_down", "drowsiness", 5.0,
             lambda t, v: _eyes(_add(v, "pitch_rel", 20.0 * _ramp(t, 0, 0.4)), 0.3)),
    Scenario("head_turn", "distraction", 4.0,
             lambda t, v: _add(v, "yaw_rel", 35.0 * _ramp(t, 0, 0.3))),
    Scenario("gaze_away", "distraction", 4.0, _gaze_away),
    Scenario("look_down", "concentration", 5.0,
             lambda t, v: _add(v, "pitch_rel", 20.0 * _ramp(t, 0, 0.4))),
    Scenario("glances", "concentration", 20.0, _glances),
)
