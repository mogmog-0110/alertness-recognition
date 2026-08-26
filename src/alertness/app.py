"""アプリ本体。ループ・キャリブレーション・キー操作をまとめる。

表示と録画は出力先(sink)に任せ、ここは流れの制御に集中する。
'q' で終了、'c' で再キャリブレーション。

画面を出さない運転（feedback.window: false）にも対応する。車載にはウィンドウもキーも
無いので、終了は SIGINT/SIGTERM で受ける。加えて Watchdog を立て、判定が流れなくなった
ことを知らせる。この装置の最悪の壊れ方は警告のしすぎではなく、黙ることなので。
"""

from __future__ import annotations

import argparse
import signal
from typing import Any

from . import factory, profiling
from .calibration.store import save_profile
from .config import load_config
from .labeling import LabelState, key_label_map
from .watchdog import Watchdog

_KEY_QUIT = ord("q")
_KEY_RECALIBRATE = ord("c")
# 検出がこの回数だけ続けて失敗したら、一時的な不調ではなく壊れていると見る。
_MAX_DETECT_FAILURES = 30


class App:
    def __init__(
        self,
        config: dict[str, Any],
        record: bool = False,
        video: str | None = None,
        label: str = "",
        guided: bool = False,
        protocol: str = "acted",
        rounds: int = 3,
        subject: str = "",
        scenario: str = "",
    ) -> None:
        self._config = config
        self._feedback = config.get("feedback", {})
        # シナリオ再生: 動画と「その時刻に何が起きているはず」を組にして流す。
        # 舞台で人が眠くなるのを待てないので、再現できる形を用意しておく。
        self._scenario = self._load_scenario(scenario) if scenario else None
        if self._scenario is not None:
            video = self._scenario.video
            record = True  # 流したそのままを採点に回せるように録っておく
        self._labels = self._make_labels(label)
        self._key_labels = key_label_map(factory.dimension_names(config))
        self._guided = self._make_guided(rounds, protocol) if guided else None
        self._cue = self._make_cue() if guided else None
        self._last_guided_key: tuple | None = None
        # ガイド時は必ず録画し、表示はアプリ側が指示画面ごと描く。
        self._source = factory.build_source(config, video, realtime=self._scenario is not None)
        self._pipeline = factory.build_pipeline(config)
        self._sinks = factory.build_sinks(
            config,
            record or guided,
            self._labels,
            window=not guided,
            subject=subject,
            source=self._source,
        )
        self._calibrator = factory.build_calibrator(config)

        calib = config.get("calibration", {})
        self._calibrating = calib.get("enabled", True)
        self._save_path = calib.get("save_path", "")
        self._gui = self._feedback.get("window", True)
        self._window_width = self._feedback.get("window_width", 0)
        # 段ごとの所要時間は明示的に頼まれたときだけ測る（fps が出ないときの切り分け用）。
        profiling.enable(self._feedback.get("profile", False))
        self._stopping = False
        self._detect_failures = 0
        self._watchdog = Watchdog(
            stall_seconds=self._feedback.get("stall_seconds", 3.0),
            repeat_seconds=self._feedback.get("stall_repeat_seconds", 5.0),
            on_stall=self._on_stall,
            on_recover=self._on_recover,
        )

    @staticmethod
    def _on_stall(silent: float) -> None:
        print(f"[異常] 判定が {silent:.1f} 秒とまっています。カメラと接続を確認してください。")

    @staticmethod
    def _on_recover(_silent: float) -> None:
        print("[復帰] 判定が再開しました。")

    def request_stop(self) -> None:
        """外から終了を頼む。画面もキーも無い運転で使う。

        ネットワーク越しの入力は、繋がっていない間フレームを待ち続ける。旗を立てるだけ
        では待ちの中にいるループがそれを読めないので、入力側の待ちも解く。
        """
        self._stopping = True
        interrupt = getattr(self._source, "interrupt", None)
        if callable(interrupt):
            interrupt()

    @staticmethod
    def _load_scenario(path: str):
        # 取り込み用の manifest をそのまま使う。「動画＋区間ごとの軸別ラベル」という
        # 形は同じなので、シナリオ専用の書式を増やす理由がない。
        from .ingest.manifest import load_manifest

        return load_manifest(path)

    def _make_labels(self, label: str) -> LabelState:
        if self._scenario is None:
            return LabelState(label)
        from .ingest.segment_label import SegmentLabelProvider

        return SegmentLabelProvider(self._scenario)

    @staticmethod
    def _make_guided(rounds: int, protocol: str):
        from .guided import PROTOCOLS, GuidedSession

        if protocol not in PROTOCOLS:
            raise ValueError(f"未知の protocol: {protocol}（{'/'.join(PROTOCOLS)} のいずれか）")
        return GuidedSession(PROTOCOLS[protocol], rounds)

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
        self._install_signal_handlers()
        self._watchdog.start()
        try:
            frames = self._source.frames()
            while not self._stopping:
                with profiling.stage("capture"):
                    frame = next(frames, None)
                if frame is None:
                    break
                self._watchdog.beat()
                with profiling.stage("observe"):
                    obs = self._observe(frame)
                if obs is None:
                    continue
                if self._calibrating:
                    with profiling.stage("output"):
                        self._calibrate(obs)
                elif self._guided is not None:
                    if self._run_guided(obs):
                        break
                else:
                    if self._scenario is not None:
                        self._labels.apply(obs.features.timestamp)
                    with profiling.stage("classify"):
                        assessment = self._pipeline.classify(obs)
                    with profiling.stage("output"):
                        self._sinks.emit(obs, assessment)
                if self._gui and self._handle_keys():
                    break
        except KeyboardInterrupt:
            pass  # Ctrl-C は正常な止め方。finally で後始末する
        finally:
            self._close()

    def _observe(self, frame: Any) -> Any:
        """1フレームを特徴量にする。単発の失敗では止まらない。

        検出器は稀に1フレームだけ落ちることがある。そこで終了すると以降ずっと無警告に
        なるので、飛ばして次のフレームへ進む。ただし連続で失敗するのは一時的な不調では
        なく壊れているので、_MAX_DETECT_FAILURES で見切って例外を上へ返す。
        """
        try:
            obs = self._pipeline.observe(frame)
        except Exception as error:  # noqa: BLE001 - 検出器側の例外型は実装依存
            self._detect_failures += 1
            if self._detect_failures >= _MAX_DETECT_FAILURES:
                raise
            print(f"[警告] フレームの解析に失敗しました（{type(error).__name__}）。次へ進みます。")
            return None
        self._detect_failures = 0
        return obs

    def _install_signal_handlers(self) -> None:
        """SIGINT / SIGTERM で終了を頼めるようにする。

        画面を出さない運転では 'q' が押せない。ハンドラを置けない環境（メインスレッド
        でない等）もあるので、置けなければ Ctrl-C の例外処理に任せる。
        """

        def handler(_signum, _frame):
            self.request_stop()

        for name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                continue

    def _run_guided(self, obs: Any) -> bool:
        guided = self._guided
        if guided is None:
            return False
        step = guided.step(obs.frame.timestamp)
        self._labels.value = step.label
        self._maybe_cue(step)
        assessment = self._pipeline.classify(obs)
        self._sinks.emit(obs, assessment)  # CSVへ記録（表示は下で行う）
        if self._gui:
            from .feedback import display, overlay

            image = overlay.render(
                obs,
                assessment,
                self._feedback.get("draw_landmarks", True),
                self._feedback.get("debug", False),
            )
            overlay.draw_guided(
                image, step.title, step.instruction, step.phase, step.remaining, step.progress
            )
            display.show(image, self._window_width)
        return step.phase == "done"

    def _calibrate(self, obs: Any) -> None:
        self._calibrator.collect(obs)
        if self._gui:
            from .feedback import display, overlay

            display.show(
                overlay.draw_calibration(obs.frame.image, self._calibrator.progress),
                self._window_width,
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
            # 別人に替わった可能性があるので、本人前提で育てた基準と履歴も捨てる。
            self._pipeline.reset_state()
            self._calibrator = factory.build_calibrator(self._config)
            self._calibrating = True
        elif key in self._key_labels:
            # 数字キーで録画ラベルを切り替える。
            self._labels.value = self._key_labels[key]
            print(f"[label] {self._labels.value or '(none)'}")
        return False

    def _close(self) -> None:
        self._watchdog.close()
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
    parser.add_argument(
        "--protocol",
        default="acted",
        help="ガイドの指示セット。acted=演技（眠気・注意逸脱）/ stress=負荷をかけて誘発。"
        "stress は 1 周 6 分半あるので --rounds 1 で足りる",
    )
    parser.add_argument("--rounds", type=int, default=3, help="ガイド収録の周回数（既定: 3）")
    parser.add_argument("--subject", default="", help="被験者ID（人ごとの評価に使う）")
    parser.add_argument(
        "--scenario",
        default="",
        help="シナリオ再生。動画と『その時刻に何が起きているはず』を組にした manifest(JSON) を"
        "実時間で流し、判定と期待ラベルを並べて出す。舞台で人が眠くなるのを待たずに済む",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    App(
        config,
        record=args.record,
        video=args.video,
        label=args.label,
        guided=args.guided,
        protocol=args.protocol,
        rounds=args.rounds,
        subject=args.subject,
        scenario=args.scenario,
    ).run()
    return 0
