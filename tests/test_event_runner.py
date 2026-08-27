"""録画CSVからイベント単位の採点を作る係のテスト。"""

from __future__ import annotations

import csv

from alertness.evaluation.event_runner import format_axis_events, score_axis_events

FIELDS = ["session_id", "timestamp", "label_drowsiness", "dim_drowsiness_level"]


def _write(path, rows, session="s1", start=0.0, step=1.0):
    """(正解ラベル, 判定レベル) の並びを CSV に書く。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, (label, level) in enumerate(rows):
            writer.writerow(
                {
                    "session_id": session,
                    "timestamp": start + i * step,
                    "label_drowsiness": label,
                    "dim_drowsiness_level": level,
                }
            )
    return str(path)


def test_a_detected_episode_reports_its_latency(tmp_path):
    # 10秒目に危険が始まり、13秒目に警告が出る。
    rows = [("none", 0)] * 10 + [("high", 0)] * 3 + [("high", 3)] * 10 + [("none", 0)] * 10
    path = _write(tmp_path / "a.csv", rows)
    score = score_axis_events([path], "drowsiness")
    assert score.truth_count == 1
    assert score.detected == 1
    assert score.median_latency == 3.0


def test_a_low_label_is_not_counted_as_danger(tmp_path):
    # low は本人も気づかない程度の兆候。警告できなくても見逃しには数えない。
    rows = [("none", 0)] * 5 + [("low", 0)] * 10 + [("none", 0)] * 5
    path = _write(tmp_path / "a.csv", rows)
    assert score_axis_events([path], "drowsiness").truth_count == 0


def test_alerts_outside_any_danger_are_false_alarms(tmp_path):
    rows = [("none", 0)] * 10 + [("none", 3)] * 5 + [("none", 0)] * 45
    path = _write(tmp_path / "a.csv", rows)
    score = score_axis_events([path], "drowsiness")
    assert score.false_alarms == 1
    assert score.false_alarms_per_hour > 0


def test_episodes_do_not_span_two_sessions(tmp_path):
    # 別々の収録を1本の時間軸として繋ぐと、切れ目が長い1区間に化ける。
    a = _write(tmp_path / "a.csv", [("high", 3)] * 10, session="s1", start=0.0)
    b = _write(tmp_path / "b.csv", [("high", 3)] * 10, session="s2", start=0.0)
    score = score_axis_events([a, b], "drowsiness")
    assert score.truth_count == 2  # 1つに繋がっていない


def test_min_level_decides_what_counts_as_an_alert(tmp_path):
    rows = [("none", 0)] * 5 + [("high", 1)] * 10 + [("none", 0)] * 5
    path = _write(tmp_path / "a.csv", rows)
    assert score_axis_events([path], "drowsiness", min_level=2).detected == 0
    assert score_axis_events([path], "drowsiness", min_level=1).detected == 1


def test_an_axis_without_labels_is_skipped(tmp_path):
    rows = [("", 0)] * 20
    path = _write(tmp_path / "a.csv", rows)
    text = format_axis_events([path], ["drowsiness"])
    assert "ありません" in text


def test_the_summary_lists_each_axis(tmp_path):
    rows = [("none", 0)] * 5 + [("high", 3)] * 10 + [("none", 0)] * 5
    path = _write(tmp_path / "a.csv", rows)
    text = format_axis_events([path], ["drowsiness"])
    assert "[drowsiness]" in text
    assert "検出遅延" in text
