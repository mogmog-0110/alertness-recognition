"""アプリ本体。ループ・キャリブレーション・キー操作をまとめる。

表示と録画は出力先(sink)に任せ、ここは流れの制御に集中する。
'q' で終了、'c' で再キャリブレーション。
"""

from __future__ import annotations

import argparse
from typing import Any

from . import factory
from .calibration.store import save_profile
from .config import load_config
from .labeling import LabelState, key_label_map

_KEY_QUIT = ord("q")
_KEY_RECALIBRATE = ord("c")


class App:
    def __init__(
        self,
        config: dict[str, Any],
        record: bool = False,
        video: str | None = None,
        label: str = "",
        guided: bool = False,
        rounds: int = 3,
        subject: str = "",
    ) -> None:
        self._config = config
        self._feedback = config.get("feedback", {})
        self._labels = LabelState(label)
        self._key_labels = key_label_map(factory.dimension_names(config))
        self._guided = self._make_guided(rounds) if guided else None
        self._cue = self._make_cue() if guided else None
        self._last_guided_key: tuple | None = None
        # ガイド時は必ず録画し、表示はアプリ側が指示画面ごと描く。
        self._source = factory.build_source(config, video)
        self._pipeline = factory.build_pipeline(config)
        self._sinks = factory.build_sinks(
            config, record or guided, self._labels, window=not guided, subject=subject
        )
        self._calibrator = factory.build_calibrator(config)

        calib = config.get("calibration", {})
        self._calibrating = calib.get("enabled", True)
        self._save_path = calib.get("save_path", "")
        self._gui = self._feedback.get("window", True)

    @staticmethod
    def _make_guided(rounds: int):
        from .guided import DEFAULT_PROMPTS, GuidedSession

        return GuidedSession(DEFAULT_PROMPTS, rounds)

    def _make_cue(self):
        from .feedback.cue import CuePlayer

        return CuePlayer(self._feedback.get("audio", True))

    def _maybe_cue(self, step: Any) -> None:
        # 区切り（準備/開始）が変わった瞬間だけ合図音を鳴らす。
        key = (step.phase, step.title)
        if key == self._last_guided_key:
            return
        self._last_guided_key = key
        if self._cue is not None and step.phase in ("ready", "hold"):
            self._cue.play("ready" if step.phase == "ready" else "go")

    def run(self) -> None:
        try:
            for frame in self._source.frames():
                obs = self._pipeline.observe(frame)
                if self._calibrating:
                    self._calibrate(obs)
                elif self._guided is not None:
                    if self._run_guided(obs):
                        break
                else:
                    self._sinks.emit(obs, self._pipeline.classify(obs))
                if self._gui and self._handle_keys():
                    break
        finally:
            self._close()

    def _run_guided(self, obs: Any) -> bool:
        step = self._guided.step(obs.frame.timestamp)
        self._labels.value = step.label
        self._maybe_cue(step)
        assessment = self._pipeline.classify(obs)
        self._sinks.emit(obs, assessment)  # CSVへ記録（表示は下で行う）
        if self._gui:
            import cv2

            from .feedback import overlay

            image = overlay.render(
                obs,
                assessment,
                self._feedback.get("draw_landmarks", True),
                self._feedback.get("debug", False),
            )
            overlay.draw_guided(
                image, step.title, step.instruction, step.phase, step.remaining, step.progress
            )
            cv2.imshow(overlay.WINDOW_NAME, image)
        return step.phase == "done"

    def _calibrate(self, obs: Any) -> None:
        self._calibrator.collect(obs)
        if self._gui:
            import cv2

            from .feedback import overlay

            cv2.imshow(
                overlay.WINDOW_NAME,
                overlay.draw_calibration(obs.frame.image, self._calibrator.progress),
            )
        if self._calibrator.progress >= 1.0:
            profile = self._calibrator.finalize()
            self._pipeline.set_profile(profile)
            if self._save_path:
                save_profile(profile, self._save_path)
            self._calibrating = False

    def _handle_keys(self) -> bool:
        import cv2

        from .feedback import overlay

        key = cv2.waitKey(1) & 0xFF
        if key == _KEY_QUIT:
            return True
        # ウィンドウの×ボタンで閉じられたら終了する。
        try:
            if cv2.getWindowProperty(overlay.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return True
        except cv2.error:
            pass
        if key == _KEY_RECALIBRATE:
            self._calibrator = factory.build_calibrator(self._config)
            self._calibrating = True
        elif key in self._key_labels:
            # 数字キーで録画ラベルを切り替える。
            self._labels.value = self._key_labels[key]
            print(f"[label] {self._labels.value or '(none)'}")
        return False

    def _close(self) -> None:
        self._sinks.close()
        self._pipeline.close()
        self._source.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="覚醒度・注意状態の認識デモ")
    parser.add_argument("--config", default="config/default.yaml", help="設定ファイル")
    parser.add_argument("--record", action="store_true", help="特徴量CSVを録画する")
    parser.add_argument("--video", default=None, help="カメラの代わりに動画ファイルを使う")
    parser.add_argument("--label", default="", help="録画時の正解ラベル（評価用）。例: drowsiness")
    parser.add_argument("--guided", action="store_true", help="ガイド付き収録モード（指示に従う）")
    parser.add_argument("--rounds", type=int, default=3, help="ガイド収録の周回数（既定: 3）")
    parser.add_argument("--subject", default="", help="被験者ID（人ごとの評価に使う）")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    App(
        config,
        record=args.record,
        video=args.video,
        label=args.label,
        guided=args.guided,
        rounds=args.rounds,
        subject=args.subject,
    ).run()
    return 0
