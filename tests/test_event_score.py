"""イベント単位の採点のテスト。

フレーム単位の accuracy では見えない2つ（検出遅延・時間あたりの誤警告）を測れているか。
"""

from __future__ import annotations

import math

from alertness.evaluation.events import (
    Episode,
    episodes_from_flags,
    format_event_score,
    score_events,
)


def _flags(pattern: str, step: float = 1.0):
    """'..##..' 形式の文字列から (時刻, フラグ) を作る。# が True。"""
    times = [i * step for i in range(len(pattern))]
    return times, [c == "#" for c in pattern]


def test_episodes_are_extracted_from_a_flag_series():
    times, flags = _flags("..###....##.")
    episodes = episodes_from_flags(times, flags)
    assert len(episodes) == 2
    assert episodes[0].start == 2.0
    assert episodes[1].start == 9.0


def test_a_briefly_broken_episode_is_treated_as_one():
    # 境界付近で1回の警告が割れると、誤警告の回数が実際より多く見える。
    times, flags = _flags("###.###")
    assert len(episodes_from_flags(times, flags)) == 1


def test_a_long_gap_separates_episodes():
    times, flags = _flags("###.....###")
    assert len(episodes_from_flags(times, flags)) == 2


def test_detection_latency_is_measured_from_the_start_of_the_danger():
    truth = [Episode(10.0, 30.0)]
    alerts = [Episode(13.0, 25.0)]
    score = score_events(truth, alerts, total_seconds=100.0)
    assert score.detected == 1
    assert score.median_latency == 3.0
    assert score.false_alarms == 0


def test_an_alert_that_starts_early_is_not_credited_with_negative_latency():
    # 負の遅延を混ぜると中央値が「早く気づけている」と読めてしまうが、実際には
    # 危険が始まる前から鳴っていた＝誤警告に近い振る舞い。
    score = score_events([Episode(10.0, 30.0)], [Episode(5.0, 25.0)], total_seconds=100.0)
    assert score.median_latency == 0.0


def test_a_missed_episode_lowers_the_detection_rate():
    truth = [Episode(10.0, 20.0), Episode(40.0, 50.0)]
    alerts = [Episode(12.0, 18.0)]
    score = score_events(truth, alerts, total_seconds=100.0)
    assert score.detection_rate == 0.5
    assert score.detected == 1


def test_false_alarms_are_counted_per_hour_of_safe_driving():
    # 実運用で最初に起きる失敗は「誤警告が多くて装置を切られる」こと。
    truth = [Episode(0.0, 60.0)]  # 危険 1 分
    alerts = [Episode(100.0, 110.0), Episode(200.0, 210.0), Episode(300.0, 310.0)]
    score = score_events(truth, alerts, total_seconds=1860.0)  # 全体 31 分
    assert score.false_alarms == 3
    assert score.safe_seconds == 1800.0  # 平常 30 分
    assert score.false_alarms_per_hour == 6.0


def test_no_detections_report_no_latency():
    score = score_events([Episode(10.0, 20.0)], [], total_seconds=100.0)
    assert math.isnan(score.median_latency)
    assert score.detection_rate == 0.0
    assert "—" in format_event_score(score)


def test_the_summary_names_all_three_numbers():
    score = score_events([Episode(10.0, 20.0)], [Episode(11.0, 19.0)], total_seconds=100.0)
    text = format_event_score(score)
    assert "検出遅延" in text
    assert "誤警告" in text
    assert "回/時" in text
