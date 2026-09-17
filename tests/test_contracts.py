"""層をまたぐデータ定義のうち、振る舞いを持つ部分のテスト。"""

from __future__ import annotations

from alertness.contracts import Assessment, Dimension, Level


def _assessment(*dims: Dimension) -> Assessment:
    return Assessment(dimensions={d.name: d for d in dims}, timestamp=0.0)


def test_headline_names_the_axis_that_holds_the_alert_level():
    # 眠気は HIGH にラッチされたまま alarm が下がりかけ、脇見は MEDIUM で alarm が上。
    # 端末は alert_level() の段と並べて軸名を出すので、段の高い軸を名指ししないと
    # 「脇見が強いです」と段と食い違う文言になる。
    assessment = _assessment(
        Dimension("drowsiness", 0.75, Level.HIGH),
        Dimension("distraction", 0.78, Level.MEDIUM),
    )
    head = assessment.headline()
    assert head is not None
    assert head.level == assessment.alert_level()
    assert head.name == "drowsiness"


def test_headline_breaks_a_level_tie_by_alarm():
    assessment = _assessment(
        Dimension("drowsiness", 0.62, Level.MEDIUM),
        Dimension("distraction", 0.70, Level.MEDIUM),
    )
    assert assessment.headline().name == "distraction"


def test_headline_uses_the_alarm_of_an_inverted_axis():
    # 集中は score が低いほど警告。score で比べると反転した軸を取り違える。
    assessment = _assessment(
        Dimension("concentration", 0.1, Level.HIGH, alert_score=0.9),
        Dimension("stress", 0.85, Level.HIGH),
    )
    assert assessment.headline().name == "concentration"


def test_headline_is_none_without_dimensions():
    assert _assessment().headline() is None
