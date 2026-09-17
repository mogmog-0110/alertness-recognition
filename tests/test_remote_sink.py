"""端末へ返す判定結果のテスト。

端末側の表示と音・振動はこの JSON だけで決まるので、段階の名前と alert の立ち方が
契約になる。
"""

from __future__ import annotations

import json

from _helpers import make_observation

from alertness.contracts import Assessment, CueResult, Dimension, Features, Level
from alertness.feedback.cadence import AlertCadence
from alertness.feedback.remote import RemoteSink


class _FakeLink:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send(self, payload: dict) -> None:
        self.sent.append(payload)


def _assessment(
    *dims: Dimension, timestamp: float = 1.0, cues: tuple[CueResult, ...] = ()
) -> Assessment:
    return Assessment(dimensions={d.name: d for d in dims}, timestamp=timestamp, cues=cues)


def _emit(sink: RemoteSink, assessment: Assessment, values: dict | None = None) -> dict:
    obs = make_observation(Features(values=values or {}, timestamp=assessment.timestamp))
    sink.emit(obs, assessment)
    return sink._link.sent[-1]  # type: ignore[attr-defined]


def test_the_level_name_matches_the_protocol():
    for level, name in ((Level.NONE, "none"), (Level.LOW, "low"), (Level.HIGH, "high")):
        link = _FakeLink()
        payload = _emit(RemoteSink(link), _assessment(Dimension("drowsiness", 0.5, level)))
        assert payload["level"] == name


def test_the_device_only_buzzes_from_medium_up():
    # LOW で鳴らすと、鳴りっぱなしになって装置ごと切られる。
    link = _FakeLink()
    sink = RemoteSink(link)
    assert _emit(sink, _assessment(Dimension("drowsiness", 0.4, Level.LOW)))["alert"] is False
    assert _emit(sink, _assessment(Dimension("drowsiness", 0.7, Level.MEDIUM)))["alert"] is True


def test_the_worst_axis_is_the_one_named():
    link = _FakeLink()
    payload = _emit(
        RemoteSink(link),
        _assessment(
            Dimension("stress", 0.6, Level.MEDIUM, alert_name="ストレス"),
            Dimension("drowsiness", 0.95, Level.HIGH, alert_name="眠気"),
        ),
    )
    assert payload["dimension"] == "眠気"
    assert payload["dimension_key"] == "drowsiness"
    assert payload["message"] == "眠気が強いです"


def test_a_medium_reads_as_a_sign_not_a_fact():
    link = _FakeLink()
    payload = _emit(
        RemoteSink(link), _assessment(Dimension("drowsiness", 0.6, Level.MEDIUM, alert_name="眠気"))
    )
    assert payload["message"] == "眠気の兆候があります"


def test_only_the_requested_features_are_sent():
    link = _FakeLink()
    sink = RemoteSink(link, features=("ear", "mar"))
    payload = _emit(
        sink,
        _assessment(Dimension("drowsiness", 0.1, Level.NONE)),
        {"ear": 0.19, "mar": 0.3, "yaw": 12.0},
    )
    assert payload["features"] == {"ear": 0.19, "mar": 0.3}


def test_unmeasured_features_are_dropped():
    # NaN は JSON で表せない。送ると端末側の復号が落ちる。
    link = _FakeLink()
    sink = RemoteSink(link, features=("ear", "hr_bpm"))
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
        RemoteSink(link), _assessment(Dimension("drowsiness", 0.1, Level.NONE)), {"ear": 0.2}
    )
    assert "features" not in payload


def test_the_axis_is_shown_in_the_language_of_the_device():
    # OpenCV の窓は日本語を描けないので軸名は英語のままだが、端末の画面は日本語で出せる。
    link = _FakeLink()
    sink = RemoteSink(link, names={"drowsiness": "眠気"})
    payload = _emit(sink, _assessment(Dimension("drowsiness", 0.95, Level.HIGH)))
    assert payload["dimension"] == "眠気"
    assert payload["dimension_key"] == "drowsiness"
    assert payload["message"] == "眠気が強いです"


def test_the_warning_name_is_what_the_table_looks_up():
    # 集中の軸は「集中」ではなく「注意散漫」として警告する。
    link = _FakeLink()
    sink = RemoteSink(link, names={"inattentive": "注意散漫"})
    dim = Dimension("concentration", 0.1, Level.HIGH, alert_score=0.9, alert_name="inattentive")
    payload = _emit(sink, _assessment(dim))
    assert payload["dimension"] == "注意散漫"
    assert payload["dimension_key"] == "inattentive"


def test_an_unmapped_axis_keeps_its_own_name():
    link = _FakeLink()
    sink = RemoteSink(link, names={"drowsiness": "眠気"})
    payload = _emit(sink, _assessment(Dimension("fatigue", 0.95, Level.HIGH)))
    assert payload["dimension"] == "fatigue"
    assert payload["dimension_key"] == "fatigue"


def test_calibration_progress_reaches_the_device() -> None:
    # キャリブ中は emit が呼ばれないので、この口が無いと端末は何も知らされない。
    link = _FakeLink()
    sink = RemoteSink(link)
    obs = make_observation(Features(values={}, timestamp=3.5))

    sink.calibrating(obs, 0.25)

    (payload,) = link.sent
    assert payload["phase"] == "calibrating"
    assert payload["progress"] == 0.25
    assert payload["timestamp"] == 3.5
    # 基準が無いままの判定で鳴らしても意味が無いので、必ず落としておく。
    assert payload["alert"] is False


def test_calibration_progress_is_clamped() -> None:
    # collect の実装によっては 1.0 を少し超えることがある。端末の進捗バーが
    # 振り切れないよう、送る側で丸める。
    link = _FakeLink()
    sink = RemoteSink(link)
    obs = make_observation(Features(values={}, timestamp=0.0))

    sink.calibrating(obs, 1.4)
    sink.calibrating(obs, -0.2)

    assert [p["progress"] for p in link.sent] == [1.0, 0.0]


def test_running_phase_is_stated_explicitly() -> None:
    # 端末は phase 省略を running とみなすが、明示しておくと受け側の分岐が
    # 「省略された」のか「running だった」のか迷わずに済む。
    link = _FakeLink()
    sink = RemoteSink(link)
    payload = _emit(sink, _assessment(Dimension(name="眠気", score=0.1, level=Level.NONE)))
    assert payload["phase"] == "running"


def test_guided_prompts_reach_the_device() -> None:
    # 運転者は PC の窓を見られない。指示が届かなければ演技のしようがない。
    link = _FakeLink()
    sink = RemoteSink(link)
    obs = make_observation(Features(values={}, timestamp=9.0))

    sink.guiding(
        obs,
        "眠い状態",
        "・まぶたを半分まで下げる",
        "hold",
        4.25,
        0.5,
        "acted_drowsiness",
    )

    (payload,) = link.sent
    assert payload["phase"] == "guided"
    assert payload["guided"]["title"] == "眠い状態"
    assert payload["guided"]["prompt_key"] == "acted_drowsiness"
    assert payload["guided"]["step"] == "hold"
    assert payload["guided"]["remaining"] == 4.2  # 小数第1位まで
    assert payload["guided"]["progress"] == 0.5
    # 指示は判定ではないので、警告を鳴らしてはいけない。
    assert payload["alert"] is False


def test_guided_progress_is_clamped() -> None:
    link = _FakeLink()
    sink = RemoteSink(link)
    obs = make_observation(Features(values={}, timestamp=0.0))
    sink.guiding(obs, "t", "i", "ready", 1.0, 1.7)
    assert link.sent[0]["guided"]["progress"] == 1.0
    assert link.sent[0]["guided"]["prompt_key"] == ""


_SOUNDS = {"drowsiness": "drowsy", "distraction": "distracted"}


def _beeps(sink: RemoteSink, dims_at: list[tuple[float, tuple[Dimension, ...]]]) -> list:
    return [_emit(sink, _assessment(*dims, timestamp=t)).get("beep") for t, dims in dims_at]


def test_the_server_decides_when_the_device_beeps():
    # 端末に間隔を持たせると、設定の間隔も HIGH での詰めも効かない。
    sink = RemoteSink(_FakeLink(), sounds=_SOUNDS, cadence=AlertCadence(5.0, 1.5, 0.7))
    medium = (Dimension("drowsiness", 0.7, Level.MEDIUM),)
    beeps = _beeps(sink, [(0.0, medium), (1.0, medium), (5.0, medium)])
    assert beeps == [{"sound": "drowsy", "level": "medium"}, None,
                     {"sound": "drowsy", "level": "medium"}]


def test_each_axis_has_its_own_sound():
    sink = RemoteSink(_FakeLink(), sounds=_SOUNDS)
    (beep,) = _beeps(sink, [(0.0, (Dimension("distraction", 0.9, Level.HIGH),))])
    assert beep == {"sound": "distracted", "level": "high"}


def test_stress_is_shown_but_never_beeps():
    # 緊張している運転者に警告音を重ねない。表示だけにする。
    sink = RemoteSink(_FakeLink(), sounds=_SOUNDS)
    payload = _emit(sink, _assessment(Dimension("stress", 0.9, Level.HIGH)))
    assert payload["alert"] is True
    assert "beep" not in payload


def test_the_louder_axis_wins_when_both_are_due():
    sink = RemoteSink(_FakeLink(), sounds=_SOUNDS)
    dims = (Dimension("drowsiness", 0.7, Level.MEDIUM), Dimension("distraction", 0.9, Level.HIGH))
    (beep,) = _beeps(sink, [(0.0, dims)])
    assert beep == {"sound": "distracted", "level": "high"}


def test_the_strongest_cue_is_sent_as_the_cause():
    # 顔を見失ったときに「眠気が強い」だけでは運転者が戸惑う。原因を添える。
    cues = (
        CueResult("eye_closure", "drowsiness", 0.4, True),
        CueResult("face_absent", "drowsiness", 1.0, True),
        CueResult("head_turn", "distraction", 1.0, True),
    )
    dim = Dimension("drowsiness", 0.95, Level.HIGH, ("eye_closure", "face_absent"))
    payload = _emit(RemoteSink(_FakeLink()), _assessment(dim, cues=cues))
    assert payload["cause"] == "face_absent"


def test_an_inverted_axis_names_its_weakest_cue():
    # 集中の軸は score が低いほど警告なので、一番低い cue が原因。
    cues = (
        CueResult("attention_buffer", "concentration", 0.1, True),
        CueResult("gaze_scanning", "concentration", 0.6, True),
    )
    dim = Dimension(
        "concentration", 0.1, Level.HIGH, ("attention_buffer", "gaze_scanning"),
        alert_score=0.9, alert_name="inattentive",
    )
    payload = _emit(RemoteSink(_FakeLink()), _assessment(dim, cues=cues))
    assert payload["cause"] == "attention_buffer"


def test_preparing_holds_back_judgements_and_restarts_the_cadence():
    link = _FakeLink()
    sink = RemoteSink(link, sounds=_SOUNDS, cadence=AlertCadence(5.0, 1.5, 0.7))
    high = (Dimension("drowsiness", 0.9, Level.HIGH),)
    assert _beeps(sink, [(0.0, high)])[0] is not None
    sink.preparing(make_observation(Features(values={}, timestamp=0.5)))
    assert link.sent[-1] == {"timestamp": 0.5, "phase": "preparing", "alert": False}
    # 準備を挟んだ後の警告は前の続きではなく、最初の 1 回としてすぐ鳴る。
    assert _beeps(sink, [(1.0, high)])[0] is not None


def test_an_axis_that_was_not_chosen_still_beeps_on_its_own_turn():
    # 同時に鳴らし時になって選ばれなかった軸を、鳴らしたことにしてはいけない。
    # 直後に重ねると 2 つの音が重なるので、前の音から最短間隔だけ空けて鳴らす。
    sink = RemoteSink(_FakeLink(), sounds=_SOUNDS, cadence=AlertCadence(5.0, 1.5, 0.7))
    both = (Dimension("drowsiness", 0.7, Level.MEDIUM), Dimension("distraction", 0.9, Level.HIGH))
    beeps = _beeps(sink, [(0.0, both), (0.5, both), (1.5, both)])
    assert beeps == [{"sound": "distracted", "level": "high"}, None,
                     {"sound": "drowsy", "level": "medium"}]
