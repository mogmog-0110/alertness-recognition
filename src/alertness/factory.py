"""設定から各アダプタを組み立てる配線層。

実装の選択（source/detector/cue/policy/sink）はすべてここに集約する。
新しい実装を足すときは、対応する registry に1行加えるだけで済む。
重い依存（mediapipe など）は、その実装が選ばれたときだけ import する。
"""

from __future__ import annotations

from typing import Any

from .classifier.classifier import CueClassifier
from .classifier.cues.attention_buffer import AttentionBufferCue
from .classifier.cues.blink import BlinkCue
from .classifier.cues.eye_closure import EyeClosureCue
from .classifier.cues.facial_tension import FacialTensionCue
from .classifier.cues.gaze_off import GazeOffCue
from .classifier.cues.gaze_scanning import GazeScanningCue
from .classifier.cues.head_down import HeadDownCue
from .classifier.cues.head_turn import HeadTurnCue
from .classifier.cues.hr_elevation import HrElevationCue
from .classifier.cues.yawn import YawnCue
from .classifier.policies.rule_based import RuleBasedPolicy
from .classifier.states import ALERT_ON, COMBINE, DimensionSpec
from .features.extractor import FaceFeatureExtractor
from .labeling import LabelState
from .pipeline import Pipeline
from .ports import Classifier, FeedbackSink, FrameSource, LandmarkDetector
from .temporal import TemporalContext

# cue 名 → 実装。新しい cue はここに登録する。
_CUE_REGISTRY = {
    "eye_closure": EyeClosureCue,
    "blink": BlinkCue,
    "yawn": YawnCue,
    "head_down": HeadDownCue,
    "head_turn": HeadTurnCue,
    "gaze_off": GazeOffCue,
    "gaze_scanning": GazeScanningCue,
    "attention_buffer": AttentionBufferCue,
    "hr_elevation": HrElevationCue,
    "facial_tension": FacialTensionCue,
}


def build_source(config: dict[str, Any], video: str | None = None) -> FrameSource:
    camera = config.get("camera", {})
    source = config.get("source", {})
    stype = "video" if video else source.get("type", "webcam")
    if stype == "webcam":
        from .sources.webcam import WebcamSource

        return WebcamSource(
            camera.get("index", 0),
            camera.get("width", 1280),
            camera.get("height", 720),
            camera.get("target_fps", 0),
            camera.get("threaded", True),
        )
    if stype == "video":
        from .sources.video_file import VideoFileSource

        path = video or source.get("path")
        if not path:
            raise ValueError("source.type=video には path（または --video）が必要です。")
        return VideoFileSource(path)
    raise ValueError(f"未知の source type: {stype}")


def build_detector(config: dict[str, Any]) -> LandmarkDetector:
    detector = config.get("detector", {})
    dtype = detector.get("type", "mediapipe")
    if dtype == "mediapipe":
        from .detection.mediapipe_detector import MediaPipeDetector

        return MediaPipeDetector(
            detector.get("model_path", "models/face_landmarker.task"),
            detector.get("max_faces", 1),
            detector.get("output_blendshapes", True),
        )
    raise ValueError(f"未知の detector type: {dtype}")


def _dimension_specs(config: dict[str, Any]) -> list[DimensionSpec]:
    specs = []
    for dim in config.get("assessment", {}).get("dimensions", []):
        alert_on = str(dim.get("alert_on", "high"))
        if alert_on not in ALERT_ON:
            raise ValueError(f"{dim['name']}: alert_on は {'/'.join(ALERT_ON)} のいずれか。")
        combine = str(dim.get("combine", "max"))
        if combine not in COMBINE:
            raise ValueError(f"{dim['name']}: combine は {'/'.join(COMBINE)} のいずれか。")
        specs.append(
            DimensionSpec(
                dim["name"],
                dict(dim.get("levels", {})),
                tuple(dim.get("cues", [])),
                alert_on,
                str(dim.get("alert_name", "")),
                combine,
            )
        )
    return specs


def build_classifier(config: dict[str, Any]) -> Classifier:
    policy_cfg = config.get("policy", {})
    ptype = policy_cfg.get("type", "rule_based")
    if ptype == "ml":
        # 学習済み model.pkl を読む判定器。cue は使わないので組み立てない。
        from .classifier.ml_based import MLClassifier, load_bundle

        return MLClassifier(
            load_bundle(policy_cfg.get("model_path", "models/model.pkl")),
            _dimension_specs(config),
        )
    if ptype != "rule_based":
        raise ValueError(f"未知の policy type: {ptype}")

    cue_cfg = config.get("cues", {})
    used: set[str] = set()
    for dim in config.get("assessment", {}).get("dimensions", []):
        used.update(dim.get("cues", []))

    cues = []
    weights: dict[str, float] = {}
    for name in sorted(used):
        if name not in _CUE_REGISTRY:
            raise ValueError(f"未知の cue: {name}")
        params = dict(cue_cfg.get(name, {}))
        weights[name] = float(params.pop("weight", 1.0))
        cues.append(_CUE_REGISTRY[name](**params))

    policy = RuleBasedPolicy(
        _dimension_specs(config), weights, policy_cfg.get("hysteresis_frames", 8)
    )
    return CueClassifier(cues, policy)


def build_rppg(config: dict[str, Any]):
    # rPPG は既定で無効。stress の特徴が要るときだけ config で有効にする。
    rcfg = config.get("rppg", {})
    if not rcfg.get("enabled", False):
        return None
    from .features.rppg import RppgEstimator

    fps = config.get("camera", {}).get("target_fps", 30)
    return RppgEstimator(
        fps=fps,
        window_seconds=rcfg.get("window_seconds", 10.0),
        min_bpm=rcfg.get("min_bpm", 42.0),
        max_bpm=rcfg.get("max_bpm", 180.0),
        hrv_min_quality=rcfg.get("hrv_min_quality", 0.25),
        hrv_enabled=rcfg.get("hrv_enabled", False),
        hrv_upsample=rcfg.get("hrv_upsample", 16),
    )


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    fps = config.get("camera", {}).get("target_fps", 30)
    temporal = TemporalContext(max_seconds=60.0, fps=fps)
    return Pipeline(
        detector=build_detector(config),
        extractor=FaceFeatureExtractor(),
        classifier=build_classifier(config),
        temporal=temporal,
        normalize_version=config.get("normalize", {}).get("version", 1),
        rppg=build_rppg(config),
    )


def dimension_names(config: dict[str, Any]) -> list[str]:
    return [d["name"] for d in config.get("assessment", {}).get("dimensions", [])]


def cue_names(config: dict[str, Any]) -> list[str]:
    used: set[str] = set()
    for dim in config.get("assessment", {}).get("dimensions", []):
        used.update(dim.get("cues", []))
    return sorted(used)


def build_sinks(
    config: dict[str, Any],
    record: bool,
    labels: LabelState,
    window: bool = True,
    subject: str = "",
) -> FeedbackSink:
    sinks: list[FeedbackSink] = []
    feedback = config.get("feedback", {})
    recorder = config.get("recorder", {})
    recording = record or recorder.get("enabled", False)

    if window and feedback.get("window", True):
        from .feedback.window_sink import OpenCvWindowSink

        sinks.append(
            OpenCvWindowSink(
                feedback.get("draw_landmarks", True),
                feedback.get("audio", True),
                feedback.get("alert_cooldown_seconds", 5.0),
                feedback.get("debug", False),
                feedback.get("sounds", {}),
                labels if recording else None,
                feedback.get("face_mesh", False),
                feedback.get("stress_meter", False),
                feedback.get("timeline", ""),
                feedback.get("timeline_seconds", 300.0),
                feedback.get("window_width", 0),
            )
        )
    if recording:
        from .feedback.csv_sink import CsvRecorderSink

        sinks.append(
            CsvRecorderSink(
                recorder.get("path", "runs"),
                dimension_names(config),
                labels,
                subject,
                cue_names(config),
            )
        )

    from .feedback.composite import CompositeSink

    return CompositeSink(sinks)


def build_calibrator(config: dict[str, Any]):
    from .calibration.calibrator import StatisticalCalibrator

    calib = config.get("calibration", {})
    fps = config.get("camera", {}).get("target_fps", 30)
    # rPPG が有効なら、心拍の安静基準も取れるようキャリブを心拍が出るまで延ばす。
    # 上限は rppg の窓長に少し余裕を足した値（窓が満ちれば心拍が出る）。
    rppg = config.get("rppg", {})
    require: tuple[str, ...] = ()
    max_seconds = 0.0
    if rppg.get("enabled", False):
        require = ("hr_bpm",)
        max_seconds = float(rppg.get("window_seconds", 20)) + 10.0
    return StatisticalCalibrator(
        calib.get("duration_seconds", 3.0), fps, require_keys=require, max_seconds=max_seconds
    )
