"""端末へ返す判定結果のテスト。

端末側の表示と音・振動はこの JSON だけで決まるので、段階の名前と alert の立ち方が
契約になる。
"""

from __future__ import annotations

import json

from _helpers import make_observation

from alertness.contracts import Assessment, Dimension, Features, Level
from alertness.feedback.iphone_ws import IPhoneSink


class _FakeLink:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send(self, payload: dict) -> None:
        self.sent.append(payload)


def _assessment(*dims: Dimension, timestamp: float = 1.0) -> Assessment:
    return Assessment(dimensions={d.name: d for d in dims}, timestamp=timestamp)


def _emit(sink: IPhoneSink, assessment: Assessment, values: dict | None = None) -> dict:
    obs = make_observation(Features(values=values or {}, timestamp=assessment.timestamp))
    sink.emit(obs, assessment)
    return sink._link.sent[-1]  # type: ignore[attr-defined]


def test_the_level_name_matches_the_protocol():
    for level, name in ((Level.NONE, "none"), (Level.LOW, "low"), (Level.HIGH, "high")):
        link = _FakeLink()
        payload = _emit(IPhoneSink(link), _assessment(Dimension("drowsiness", 0.5, level)))
        assert payload["level"] == name


def test_the_device_only_buzzes_from_medium_up():
    # LOW で鳴らすと、鳴りっぱなしになって装置ごと切られる。
    link = _FakeLink()
    sink = IPhoneSink(link)
    assert _emit(sink, _assessment(Dimension("drowsiness", 0.4, Level.LOW)))["alert"] is False
    assert _emit(sink, _assessment(Dimension("drowsiness", 0.7, Level.MEDIUM)))["alert"] is True


def test_the_worst_axis_is_the_one_named():
    link = _FakeLink()
    payload = _emit(
        IPhoneSink(link),
        _assessment(
            Dimension("stress", 0.6, Level.MEDIUM, alert_name="ストレス"),
            Dimension("drowsiness", 0.95, Level.HIGH, alert_name="眠気"),
        ),
    )
    assert payload["dimension"] == "眠気"
    assert payload["message"] == "眠気が強いです"


def test_a_medium_reads_as_a_sign_not_a_fact():
    link = _FakeLink()
    payload = _emit(
        IPhoneSink(link), _assessment(Dimension("drowsiness", 0.6, Level.MEDIUM, alert_name="眠気"))
    )
    assert payload["message"] == "眠気の兆候があります"


def test_only_the_requested_features_are_sent():
    link = _FakeLink()
    sink = IPhoneSink(link, features=("ear", "mar"))
    payload = _emit(
        sink,
        _assessment(Dimension("drowsiness", 0.1, Level.NONE)),
        {"ear": 0.19, "mar": 0.3, "yaw": 12.0},
    )
    assert payload["features"] == {"ear": 0.19, "mar": 0.3}


def test_unmeasured_features_are_dropped():
    # NaN は JSON で表せない。送ると端末側の復号が落ちる。
    link = _FakeLink()
    sink = IPhoneSink(link, features=("ear", "hr_bpm"))
    payload = _emit(
        sink,
        _assessment(Dimension("drowsiness", 0.1, Level.NONE)),
        {"ear": 0.2, "hr_bpm": float("nan")},
    )
    assert payload["features"] == {"ear": 0.2}
    json.loads(json.dumps(payload))  # そのまま送れる形になっている


def test_nothing_is_asked_for_when_no_features_are_configured():
    link = _FakeLink()
    payload = _emit(
        IPhoneSink(link), _assessment(Dimension("drowsiness", 0.1, Level.NONE)), {"ear": 0.2}
    )
    assert "features" not in payload


def test_the_axis_is_shown_in_the_language_of_the_device():
    # OpenCV の窓は日本語を描けないので軸名は英語のままだが、端末の画面は日本語で出せる。
    link = _FakeLink()
    sink = IPhoneSink(link, names={"drowsiness": "眠気"})
    payload = _emit(sink, _assessment(Dimension("drowsiness", 0.95, Level.HIGH)))
    assert payload["dimension"] == "眠気"
    assert payload["message"] == "眠気が強いです"


def test_the_warning_name_is_what_the_table_looks_up():
    # 集中の軸は「集中」ではなく「注意散漫」として警告する。
    link = _FakeLink()
    sink = IPhoneSink(link, names={"inattentive": "注意散漫"})
    dim = Dimension("concentration", 0.1, Level.HIGH, alert_score=0.9, alert_name="inattentive")
    payload = _emit(sink, _assessment(dim))
    assert payload["dimension"] == "注意散漫"


def test_an_unmapped_axis_keeps_its_own_name():
    link = _FakeLink()
    sink = IPhoneSink(link, names={"drowsiness": "眠気"})
    payload = _emit(sink, _assessment(Dimension("fatigue", 0.95, Level.HIGH)))
    assert payload["dimension"] == "fatigue"
