"""EDA から覚醒度・段階ラベルを作る純関数のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from alertness.bio import relative_arousal, stage_from_arousal, subject_scale, tonic_windows


def test_tonic_windows_splits_and_takes_median():
    # 4Hz、2秒窓 → 8サンプルずつ。定数区間なら中央値はその値。
    signal = [1.0] * 8 + [3.0] * 8
    windows = tonic_windows(signal, fs=4.0, window_seconds=2.0)

    assert len(windows) == 2
    assert windows[0] == (1.0, 1.0)  # 中央時刻1秒、SCL 1.0
    assert windows[1] == (3.0, 3.0)


def test_tonic_windows_ignores_spikes_via_median():
    # 急峻な発汗応答(SCR)が1点あっても、中央値なら tonic 水準に引かれない。
    signal = [1.0, 1.0, 9.0, 1.0]
    assert tonic_windows(signal, fs=4.0, window_seconds=1.0)[0][1] == 1.0


def test_subject_scale_uses_rest_baseline_and_dynamic_range():
    rest = [(0.0, 0.2), (1.0, 0.2)]
    every = [(0.0, 0.2), (1.0, 0.2), (2.0, 1.2)]  # 上がった窓を含む
    baseline, spread = subject_scale(rest, every)

    assert baseline == 0.2  # 安静の中央値
    assert spread > 0  # P10-P90 の幅


def test_subject_scale_none_without_windows():
    assert subject_scale([], []) is None


def test_relative_arousal_zero_at_baseline():
    assert relative_arousal(0.2, baseline=0.2, spread=1.0) == 0.0
    assert relative_arousal(1.2, baseline=0.2, spread=1.0) == pytest.approx(1.0)


def test_stage_from_arousal_ascending():
    thresholds = (0.15, 0.40, 0.70)
    assert stage_from_arousal(0.0, thresholds) == "none"
    assert stage_from_arousal(0.3, thresholds) == "low"
    assert stage_from_arousal(0.5, thresholds) == "medium"
    assert stage_from_arousal(0.9, thresholds) == "high"


def test_stage_from_arousal_rejects_descending_thresholds():
    with pytest.raises(ValueError, match="昇順"):
        stage_from_arousal(0.5, (0.7, 0.4, 0.15))


def test_stage_from_arousal_checks_threshold_count():
    with pytest.raises(ValueError):
        stage_from_arousal(0.5, (0.5,))  # levels は4段なので3つ要る


def test_non_responder_does_not_divide_by_zero():
    # EDA が全く動かない人でも spread に下限があるので発散しない。
    flat = [(float(i), 0.1) for i in range(5)]
    baseline, spread = subject_scale(flat, flat)
    assert spread > 0
    assert np.isfinite(relative_arousal(0.1, baseline, spread))
