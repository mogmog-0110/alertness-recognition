"""共通ingestのCSV FPS設定、バッチ事前検査、CLI報告を検証する。"""

from __future__ import annotations

from pathlib import Path

import pytest

from alertness.ingest import cli
from alertness.ingest.manifest import ClipManifest
from alertness.ingest.runner import IngestBatchResult, IngestSkip, run_ingest, run_ingest_batch


def _manifest(video: str) -> ClipManifest:
    return ClipManifest(video, "subject", "", ())


def test_auto_uses_batch_minimum_and_passes_it_to_every_ingest(monkeypatch, tmp_path) -> None:
    fps_by_video = {"30.mp4": 30.0, "15.mp4": 15.0}
    calls: list[tuple[str, float]] = []

    monkeypatch.setattr(
        "alertness.sources.video_file.probe_video_fps", lambda video: fps_by_video[video]
    )

    def fake_run(config, manifest, out_base, *, csv_fps=None):
        calls.append((manifest.video, csv_fps))
        return Path(out_base) / manifest.video

    monkeypatch.setattr("alertness.ingest.runner.run_ingest", fake_run)

    result = run_ingest_batch(
        {"ingest": {"csv_fps": "auto"}},
        [_manifest("30.mp4"), _manifest("15.mp4")],
        tmp_path,
    )

    assert result.csv_fps == 15.0
    assert calls == [("30.mp4", 15.0), ("15.mp4", 15.0)]
    assert result.skipped == ()


def test_fixed_fps_skips_lower_and_unreadable_videos(monkeypatch, tmp_path) -> None:
    def fake_probe(video: str) -> float:
        if video == "broken.mp4":
            raise ValueError("FPS不明")
        return {"30.mp4": 30.0, "10.mp4": 10.0}[video]

    calls: list[str] = []
    monkeypatch.setattr("alertness.sources.video_file.probe_video_fps", fake_probe)
    monkeypatch.setattr(
        "alertness.ingest.runner.run_ingest",
        lambda config, manifest, out_base, *, csv_fps=None: calls.append(manifest.video)
        or Path(out_base) / manifest.video,
    )

    result = run_ingest_batch(
        {"ingest": {"csv_fps": 15}},
        [_manifest("30.mp4"), _manifest("10.mp4"), _manifest("broken.mp4")],
        tmp_path,
    )

    assert result.csv_fps == 15.0
    assert calls == ["30.mp4"]
    assert [item.video for item in result.skipped] == ["10.mp4", "broken.mp4"]
    assert "以下" in result.skipped[0].reason
    assert result.skipped[1].reason == "FPS不明"


def test_single_ingest_uses_same_effective_fps_for_both_passes_and_pipeline(
    monkeypatch, tmp_path
) -> None:
    seen: dict[str, list[float]] = {"calibrate": [], "source": [], "pipeline": []}

    class Pipeline:
        def set_profile(self, _profile) -> None:
            pass

        def close(self) -> None:
            pass

    class Source:
        def frames(self):
            return iter(())

        def close(self) -> None:
            pass

    class Sink:
        def __init__(self, *_args) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr("alertness.sources.video_file.probe_video_fps", lambda _video: 30.0)
    monkeypatch.setattr(
        "alertness.ingest.runner._calibrate",
        lambda _config, _manifest, fps: seen["calibrate"].append(fps) or object(),
    )
    monkeypatch.setattr(
        "alertness.ingest.runner._video_source",
        lambda _video, fps: seen["source"].append(fps) or Source(),
    )
    monkeypatch.setattr(
        "alertness.factory.build_pipeline",
        lambda _config, fps=None: seen["pipeline"].append(fps) or Pipeline(),
    )
    monkeypatch.setattr("alertness.factory.dimension_names", lambda _config: [])
    monkeypatch.setattr("alertness.factory.cue_names", lambda _config: [])
    monkeypatch.setattr("alertness.feedback.csv_sink.CsvRecorderSink", Sink)

    run_ingest({}, _manifest("30.mp4"), tmp_path, csv_fps=15.0)

    assert seen == {"calibrate": [15.0], "source": [15.0], "pipeline": [15.0]}


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), "minimum", True])
def test_invalid_csv_fps_fails_before_probing(monkeypatch, value) -> None:
    probed: list[str] = []
    monkeypatch.setattr(
        "alertness.sources.video_file.probe_video_fps",
        lambda video: probed.append(video) or 15.0,
    )

    with pytest.raises(ValueError, match="ingest.csv_fps"):
        run_ingest_batch({"ingest": {"csv_fps": value}}, [_manifest("v.mp4")])
    assert probed == []


def test_cli_returns_one_and_reports_when_all_videos_are_skipped(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "load_config", lambda _path: {})
    monkeypatch.setattr(cli, "manifests_from", lambda _path: iter([_manifest("bad.mp4")]))
    monkeypatch.setattr(
        cli,
        "run_ingest_batch",
        lambda *_args: IngestBatchResult(None, (), (IngestSkip("bad.mp4", "FPS不明"),)),
    )

    assert cli.main(["--manifests", "manifests"]) == 1
    output = capsys.readouterr().out
    assert "SKIP bad.mp4: FPS不明" in output
    assert "取り込む動画がありませんでした" in output
