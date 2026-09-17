"""実機の収録で書いた期待を確かめる仕組みのテスト。

収録そのもの（runs/regression/）は顔の動きの記録なので git に入れない。手元にあれば、
それも流して期待どおりかを確かめる。
"""

from __future__ import annotations

import os

import pytest

from alertness.config import load_config
from alertness.contracts import Assessment, Dimension, Level
from alertness.evaluation import regression
from alertness.evaluation.benchmark import Sample

_SPEC = os.path.join("runs", "regression", "phone.json")


def _samples(levels: list[tuple[float, Level, Level]]) -> list[Sample]:
    return [
        Sample(t, Assessment(
            {
                "drowsiness": Dimension("drowsiness", 0.9, drowsy),
                "distraction": Dimension("distraction", 0.8, distracted),
            },
            t,
        ))
        for t, drowsy, distracted in levels
    ]


def _check(samples, **expectation):
    return regression._check(samples, expectation)


def test_alert_quiet_and_headline_are_checked():
    samples = _samples([
        (0.0, Level.NONE, Level.NONE),
        (1.0, Level.HIGH, Level.NONE),
        (2.0, Level.NONE, Level.HIGH),
        (3.0, Level.NONE, Level.NONE),
    ])
    drowsy = {"dimension": "drowsiness"}
    assert _check(samples, kind="alert", **drowsy, **{"from": 0.5, "to": 1.5})[0]
    assert not _check(samples, kind="quiet", **drowsy, **{"from": 0.5, "to": 1.5})[0]
    assert _check(samples, kind="quiet", **drowsy, **{"from": 2.5, "to": 3.0})[0]
    assert _check(samples, kind="headline", dimension="distraction", at=2.0)[0]
    passed, observed = _check(samples, kind="headline", dimension="distraction", at=3.0)
    assert not passed
    assert "-" in observed


@pytest.mark.skipif(not os.path.isfile(_SPEC), reason="実機の収録が手元に無い")
def test_phone_recordings_are_judged_as_expected():
    config = load_config("config/browser.yaml")
    failures = [
        f"{o.csv} {o.expectation.get('note', '')}: {o.observed}"
        for csv_path, expectations in regression.load(_SPEC)
        for o in regression.check_recording(csv_path, expectations, config)
        if not o.passed
    ]
    assert not failures, "\n".join(failures)
