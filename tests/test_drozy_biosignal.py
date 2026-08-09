from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import examples.convert_drozy as convert_drozy
from alertness.bio.psg import (
    PsgFeature,
    PsgRecording,
    extract_psg_features,
    read_psg,
    resolve_channels,
)
from alertness.bio.pvt import (
    PvtSample,
    PvtSummary,
    impairment_from_baseline,
    read_pvt,
    summarize_pvt,
    summarize_pvt_windows,
)
from alertness.calibration.baseline import fit_baseline, normalize_features
from alertness.classifier.lod import calibrate_thresholds
from alertness.ingest.manifest import from_dict
from alertness.temporal import smooth_lod_segments


def _feature(timestamp: float, scale: float) -> PsgFeature:
    return PsgFeature(
        timestamp=timestamp,
        theta=scale,
        alpha=scale * 0.8,
        beta=2.0 - scale * 0.1,
        di=scale * 0.5,
        sem=scale * 0.3,
        blink_duration=scale * 0.1,
        microsleep_duration=scale * 0.05,
        valid=True,
        eeg_channels={},
    )


def test_resolve_channels_accepts_drozy_aliases_and_rejects_missing() -> None:
    labels = ["EEG Fz-A1", "Cz", "C3", "C4", "Pz", "HEOG", "VEOG"]
    aliases = {
        "fz": ["EEG Fz-A1"],
        "cz": ["Cz"],
        "c3": ["C3"],
        "c4": ["C4"],
        "pz": ["Pz"],
        "horizontal_eog": ["HEOG"],
        "vertical_eog": ["VEOG"],
    }

    resolved = resolve_channels(labels, aliases)

    assert resolved["fz"] == 0
    assert resolved["vertical_eog"] == 6
    with pytest.raises(ValueError, match="vertical_eog"):
        resolve_channels(labels[:-1], aliases)


def test_read_psg_reads_named_channels_from_real_edf_container(tmp_path) -> None:
    pyedflib = pytest.importorskip("pyedflib")
    sample_rate = 128
    samples = np.arange(sample_rate * 2) / sample_rate
    labels = ["Fz", "Cz", "C3", "C4", "Pz", "HEOG", "VEOG"]
    signals = [20.0 * np.sin(2 * np.pi * (index + 1) * samples) for index in range(7)]
    headers = [
        pyedflib.highlevel.make_signal_header(label, sample_frequency=sample_rate)
        for label in labels
    ]
    path = tmp_path / "1-1.edf"
    assert pyedflib.highlevel.write_edf(str(path), signals, headers)

    recording = read_psg(path)

    assert set(recording.signals) == {
        "fz",
        "cz",
        "c3",
        "c4",
        "pz",
        "horizontal_eog",
        "vertical_eog",
    }
    assert recording.sample_rates["fz"] == sample_rate
    assert recording.duration_seconds == pytest.approx(2.0)


def test_extract_psg_features_uses_band_power_and_window_centers() -> None:
    sample_rate = 128.0
    seconds = 12
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    eeg = np.sin(2 * np.pi * 6 * t) + 0.25 * np.sin(2 * np.pi * 18 * t)
    horizontal = np.sin(2 * np.pi * 0.5 * t)
    vertical = np.zeros_like(t)
    vertical[int(5 * sample_rate) : int(5.6 * sample_rate)] = 4.0
    names = ("fz", "cz", "c3", "c4", "pz")
    recording = PsgRecording(
        signals={
            **{name: eeg for name in names},
            "horizontal_eog": horizontal,
            "vertical_eog": vertical,
        },
        sample_rates={
            **{name: sample_rate for name in names},
            "horizontal_eog": sample_rate,
            "vertical_eog": sample_rate,
        },
        source_labels={name: name for name in (*names, "horizontal_eog", "vertical_eog")},
    )

    features = extract_psg_features(recording, window_seconds=10.0, stride_seconds=1.0)

    assert [item.timestamp for item in features] == [5.0, 6.0, 7.0]
    assert all(item.valid for item in features)
    assert features[0].theta > features[0].beta
    assert features[0].microsleep_duration >= 0.5


def test_baseline_normalization_uses_pvt1_and_omits_zero_variance() -> None:
    pvt1 = [_feature(float(index), float(index + 1)) for index in range(4)]
    baseline = fit_baseline(pvt1, baseline_seconds=10.0)
    normalized = normalize_features([_feature(5.0, 6.0)], baseline)

    assert baseline.sample_count == 4
    assert normalized[0]["theta"] > 0
    assert "beta" in normalized[0]


def test_read_pvt_parses_official_drozy_timestamp_pairs(tmp_path) -> None:
    path = tmp_path / "1-1.csv"
    path.write_text(
        "\ufeff2014-11-26_10.08.39.274\n\n"
        "2014-11-26_10.08.45.613;2014-11-26_10.08.45.926\n"
        "2014-11-26_10.08.48.935;2014-11-26_10.08.49.220\n"
        "2014-11-26_10.08.55.611;2014-11-26_10.08.55.973\n",
        encoding="utf-8",
    )

    samples = read_pvt(path)

    assert [sample.timestamp_seconds for sample in samples] == pytest.approx(
        [6.339, 9.661, 16.337]
    )
    assert [sample.reaction_ms for sample in samples] == pytest.approx([313.0, 285.0, 362.0])


@pytest.mark.parametrize(
    ("text", "line_number", "message"),
    [
        (
            "2014-11-26_10.08.39.274\n2014-11-26_10.08.45.613\n",
            2,
            "2列",
        ),
        (
            "2014-11-26_10.08.39.274\n"
            "2014-11-26_10.08.45.613;2014-11-26_10.08.45.926;extra\n",
            2,
            "2列",
        ),
        (
            "2014-11-26_10.08.39.274\nnot-a-time;2014-11-26_10.08.45.926\n",
            2,
            "解析できません",
        ),
        (
            "2014-11-26_10.08.45.000\n"
            "2014-11-26_10.08.44.000;2014-11-26_10.08.44.250\n",
            2,
            "試験開始前",
        ),
        (
            "2014-11-26_10.08.39.274\n"
            "2014-11-26_10.08.45.613;2014-11-26_10.08.45.926\n"
            "2014-11-26_10.08.44.613;2014-11-26_10.08.44.926\n",
            3,
            "単調増加",
        ),
        (
            "2014-11-26_10.08.39.274\n"
            "2014-11-26_10.08.45.613;2014-11-26_10.08.45.500\n",
            2,
            "刺激時刻より前",
        ),
    ],
)
def test_read_pvt_rejects_malformed_rows_with_line_number(
    tmp_path, text: str, line_number: int, message: str
) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message) as error:
        read_pvt(path)

    assert f"{path}:{line_number}" in str(error.value)


def test_read_pvt_handles_minute_rollover(tmp_path) -> None:
    path = tmp_path / "rollover.csv"
    path.write_text(
        "2014-11-26_10.08.59.900\n"
        "2014-11-26_10.08.59.950;2014-11-26_10.09.00.263\n",
        encoding="utf-8",
    )

    sample = read_pvt(path)[0]

    assert sample.timestamp_seconds == pytest.approx(0.05)
    assert sample.reaction_ms == pytest.approx(313.0)


def test_pvt_summary_discards_false_starts_and_lapses_from_mean() -> None:
    summary = summarize_pvt(
        [PvtSample(80.0), PvtSample(250.0), PvtSample(500.0), PvtSample(750.0)]
    )

    assert summary.mean_reaction_ms == 250.0
    assert summary.false_start_count == 1
    assert summary.normal_count == 1
    assert summary.lapse_count == 2
    assert summary.valid_count == 3
    assert summary.lapse_rate == pytest.approx(2 / 3)


def test_pvt_is_aggregated_into_twenty_second_windows() -> None:
    samples = [
        PvtSample(250.0, 1.0),
        PvtSample(550.0, 12.0),
        PvtSample(300.0, 20.0),
        PvtSample(700.0, 40.0),
    ]

    windows = summarize_pvt_windows(samples, window_seconds=20.0)

    assert [timestamp for timestamp, _summary in windows] == [10.0, 30.0, 50.0]
    assert windows[0][1].lapse_rate == 0.5
    assert windows[1][1].mean_reaction_ms == 300.0
    assert windows[2][1].mean_reaction_ms is None
    assert windows[2][1].lapse_count == 1


def test_pvt_impairment_uses_session_summary_and_pvt1_is_zero() -> None:
    baseline = PvtSummary(300.0, 0.1, 10, 1, 9, 1)
    slower = PvtSummary(360.0, 0.1, 10, 0, 9, 1)
    more_lapses = PvtSummary(300.0, 0.3, 10, 0, 7, 3)

    unchanged = impairment_from_baseline(baseline, baseline)
    slower_impairment = impairment_from_baseline(slower, baseline)
    lapse_impairment = impairment_from_baseline(more_lapses, baseline)

    assert unchanged == 0.0
    assert slower_impairment > 0.0
    assert lapse_impairment > 0.0
    base_thresholds = calibrate_thresholds((20.0, 50.0, 75.0), unchanged)
    assert calibrate_thresholds((20.0, 50.0, 75.0), slower_impairment) < base_thresholds
    assert calibrate_thresholds((20.0, 50.0, 75.0), lapse_impairment) < base_thresholds
    with pytest.raises(ValueError, match="PVT1に通常反応がなく"):
        impairment_from_baseline(
            PvtSummary(None, 1.0, 10, 0, 0, 10),
            PvtSummary(None, 1.0, 10, 0, 0, 10),
        )


def test_kss_zero_is_missing_and_sessions_are_typed(tmp_path) -> None:
    root = tmp_path / "DROZY"
    for name in ("psg", "pvt-rt", "timestamps", "videos_i8"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "KSS.txt").write_text("0 5 9\n", encoding="utf-8")
    (root / "psg" / "1-1.edf").touch()
    (root / "pvt-rt" / "1-1.csv").touch()
    (root / "pvt-rt" / "._2-3.csv").touch()
    (root / "timestamps" / "1-1.txt").touch()
    (root / "videos_i8" / "1-1.mp4").touch()

    sessions = convert_drozy.discover_sessions(root)

    assert len(sessions) == 1
    assert sessions[0].context == "PVT1"
    assert sessions[0].kss is None


def test_smooth_segments_are_continuous_and_remove_short_transition() -> None:
    scores = [10.0] * 6 + [90.0] * 2 + [10.0] * 7
    segments = smooth_lod_segments(
        scores,
        thresholds=(20.0, 50.0, 75.0),
        duration_seconds=15.0,
        median_seconds=1.0,
        min_duration_seconds=5.0,
    )

    assert segments == [{"start": 0.0, "end": 15.0, "label": "none"}]


def test_build_manifest_roundtrips_with_only_drowsiness_axis(tmp_path) -> None:
    video = tmp_path / "1-1.mp4"
    session = convert_drozy.DrozySession(
        subject="1",
        test=1,
        session_id="1-1",
        video=video,
        psg=tmp_path / "1-1.edf",
        pvt=tmp_path / "1-1.csv",
        timestamps=tmp_path / "1-1.txt",
        kss=4.0,
    )
    features = tuple(_feature(float(index + 5), float(index + 1)) for index in range(12))
    analysis = convert_drozy.SessionAnalysis(
        session=session,
        features=features,
        duration_seconds=20.0,
        pvt_summary=PvtSummary(300.0, 0.0, 10, 0, 10, 0),
        pvt_windows=(),
    )
    baseline = fit_baseline(features[:5], baseline_seconds=20.0)
    config = {
        "drozy": {
            "lod": {"thresholds": [20.0, 50.0, 75.0]},
            "temporal": {"median_seconds": 5.0, "min_duration_seconds": 5.0},
        }
    }

    result = convert_drozy.build_manifest_for_session(
        analysis, baseline, analysis.pvt_summary, config
    )
    manifest = from_dict(json.loads(json.dumps(result.manifest)))

    assert manifest.subject == "drozy_1"
    assert manifest.labels_at(1.0).keys() == {"drowsiness"}


_DROZY_ROOT = Path(os.environ.get("DROZY_ROOT", "data/DROZY"))


@pytest.mark.skipif(not (_DROZY_ROOT / "psg").is_dir(), reason="実DROZYデータがない")
def test_real_drozy_smoke_when_dataset_is_available() -> None:
    sessions = convert_drozy.discover_sessions(_DROZY_ROOT)

    assert len(sessions) == 36
    for session in sessions:
        assert session.pvt is not None
        summary = summarize_pvt(read_pvt(session.pvt))
        assert summary.valid_count > 0

