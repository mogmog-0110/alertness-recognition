"""走査による集中の手がかりのテスト。"""

from __future__ import annotations

from _helpers import FakeHistory, make_observation

from alertness.classifier.cues.gaze_scanning import GazeScanningCue
from alertness.contracts import Features


def _frames(offsets, seconds: float = 30.0):
    """offsets を繰り返して gaze_off の系列を作る。"""
    n = int(seconds * 10)
    return [
        Features({"gaze_off": offsets[i % len(offsets)], "yaw_rel": 0.0}, i * 0.1) for i in range(n)
    ]


def _last(cue, frames):
    return cue.evaluate(make_observation(frames[-1], FakeHistory(frames)))


def test_scanning_full_score_when_gaze_moves():
    # 中心と周辺を行き来している＝周囲を走査している。
    result = _last(GazeScanningCue(), _frames([0.005, 0.05, 0.01, 0.08, 0.02, 0.06]))
    assert result.score >= 0.9
    assert not result.active


def test_scanning_drops_when_gaze_is_frozen():
    # 一点に貼りついている。PRC 100% で走査していない＝認知的な注意逸脱の兆候。
    result = _last(GazeScanningCue(), _frames([0.001, 0.0011, 0.0009]))
    assert result.score <= 0.1
    assert result.active
    assert "PRC 100%" in result.detail


def test_scanning_does_not_judge_short_history():
    # 起動直後（履歴3秒）は判定しない。数十秒の窓で見る指標なので誤警告になる。
    result = _last(GazeScanningCue(min_window=10.0), _frames([0.001], seconds=3.0))
    assert result.score == 1.0
    assert not result.valid


def test_scanning_ignores_missing_gaze():
    frames = [Features({"ear": 0.3}, i * 0.1) for i in range(300)]
    result = _last(GazeScanningCue(), frames)
    assert not result.valid


def test_scanning_and_buffer_cover_opposite_failures():
    # 貼りつき（走査なし）でもバッファは満タンのまま。両方見て初めて区別できる。
    from alertness.classifier.cues.attention_buffer import AttentionBufferCue

    frames = _frames([0.001, 0.0011, 0.0009])
    buffer_cue = AttentionBufferCue()
    for i in range(0, len(frames), 5):
        buffer_result = buffer_cue.evaluate(
            make_observation(frames[i], FakeHistory(frames[: i + 1]))
        )
    assert buffer_result.score >= 1.0  # 目は対象にある
    assert _last(GazeScanningCue(), frames).score <= 0.1  # だが走査していない
