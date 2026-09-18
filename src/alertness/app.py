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

from . import factory, log, profiling
from .calibration.store import save_profile
from .config import load_config
from .labeling import LabelState, key_label_map
from .sources.remote import NEW_SESSION
from .watchdog import Watchdog

_KEY_QUIT = ord("q")
_KEY_RECALIBRATE = ord("c")
# 検出がこの回数だけ続けて失敗したら、一時的な不調ではなく壊れていると見る。
_MAX_DETECT_FAILURES = 30
# 端末がページを開き直してから、測り直しの命令が来なくても自分で測り始めるまでの秒数。
# 端末は構え直しを待って 4 秒後に命令を送るので、それより長く取る。
_SESSION_CALIBRATE_FALLBACK_SECONDS = 8.0


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
        # 端末が開き直してから、測り始めるまで判定を出さずに待っている間 True。
        self._preparing = False
        self._preparing_since: float | None = None
        self._stall_reported = False
        self._watchdog = Watchdog(
            stall_seconds=self._feedback.get("stall_seconds", 3.0),
            repeat_seconds=self._feedback.get("stall_repeat_seconds", 5.0),
            on_stall=self._on_stall,
            on_recover=self._on_recover,
        )

    def _on_stall(self, silent: float) -> None:
        # 端末が繋がっていないのは待ち受け中の正常な姿で、操作者が直すものではない。
        if not getattr(self._source, "connected", True):
            return
        self._stall_reported = True
        print(f"[異常] 判定が {silent:.1f} 秒とまっています。カメラと接続を確認してください。")

    def _on_recover(self, _silent: float) -> None:
        if not self._stall_reported:
            return
        self._stall_reported = False
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
        try:
            frames = self._source.frames()
            while not self._stopping:
                with profiling.stage("capture"):
                    frame = next(frames, None)
                if frame is None:
                    break
                # 最初の 1 枚が来てから見張る。端末が繋ぐ前の沈黙は異常ではない。
                self._watchdog.start()
                self._watchdog.beat()
                self._handle_remote_commands()
                with profiling.stage("observe"):
                    obs = self._observe(frame)
                if obs is None:
                    continue
                if self._preparing:
                    self._wait_for_calibration(obs)
                elif self._calibrating:
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
            if self._stopping:
                # 2 回目は待たない。検出器の中で固まっていると旗を読みに戻れない。
                raise KeyboardInterrupt
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
        # 端末を車載位置に置いた運転者は PC の窓を見られないので、指示も送る。
        guiding = getattr(self._sinks, "guiding", None)
        if callable(guiding):
            guiding(
                obs, step.title, step.instruction,
                step.phase, step.remaining, step.progress, step.prompt_key,
            )
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
        # 手元に窓が無い構成 (端末が離れている) でも進捗が伝わるようにする。
        notify = getattr(self._sinks, "calibrating", None)
        if callable(notify):
            waiting = getattr(self._calibrator, "waiting_for", "")
            expected = float(getattr(self._calibrator, "expected_seconds", 0.0))
            notify(obs, self._calibrator.progress, waiting, expected)
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
            self._recalibrate()
        elif key in self._key_labels:
            # 数字キーで録画ラベルを切り替える。
            self._labels.value = self._key_labels[key]
            print(f"[label] {self._labels.value or '(none)'}")
        return False

    def _handle_remote_commands(self) -> None:
        """端末から届いた操作を反映する。source が対応していなければ何もしない。

        フレームの処理ループから呼ぶので、**映像が流れていない間は処理されない**。
        端末はカメラを開始してから接続するので実害は無いが、映像を止めた状態で
        命令だけ送っても効かない。
        """
        take = getattr(self._source, "take_commands", None)
        if not callable(take):
            return
        for command in take():
            if command == "recalibrate":
                log.detail("[remote] 端末から再キャリブを受け取りました。")
                self._recalibrate()
            elif command == NEW_SESSION:
                log.detail("[remote] 端末がページを開き直しました。")
                self._start_session()
            else:
                log.detail(f"[remote] 未知の命令を無視しました: {command}")

    def _start_session(self) -> None:
        """開き直した端末の測り直しを待つ。それまで判定を出さない。

        前の人の基準のまま判定すると、構え直しの間に顔が外れただけで眠気や
        注意散漫の警告が鳴る。基準も履歴も捨て、端末が構え終わって命令を
        送ってくるまで「準備中」を返す。
        """
        self._pipeline.reset_state()
        self._calibrating = False
        self._preparing = True
        self._preparing_since = None

    def _wait_for_calibration(self, obs: Any) -> None:
        now = obs.features.timestamp
        if self._preparing_since is None:
            self._preparing_since = now
        notify = getattr(self._sinks, "preparing", None)
        if callable(notify):
            notify(obs)
        if now - self._preparing_since >= _SESSION_CALIBRATE_FALLBACK_SECONDS:
            self._recalibrate()  # 命令が届かない古いページでも測り始める

    def _recalibrate(self) -> None:
        # 別人に替わった可能性があるので、本人前提で育てた基準と履歴も捨てる。
        self._pipeline.reset_state()
        self._calibrator = factory.build_calibrator(self._config)
        self._calibrating = True
        self._preparing = False

    def _close(self) -> None:
        self._watchdog.close()
        self._sinks.close()
        self._pipeline.close()
        self._source.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="接続の出入り・端末からの命令・MediaPipe の内部ログも出す"
        "（繋がらないときの切り分け用）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args, load_config(args.config))


def run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    log.set_verbose(args.verbose)
    if _wants_multi_device(config, args):
        from .multi_device import run as run_multi_device

        return run_multi_device(config)
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


def _wants_multi_device(config: dict[str, Any], args: argparse.Namespace) -> bool:
    """複数台対応が意味を持つ構成か。

    ガイド収録・シナリオ再生・動画ファイルの読み込みは、いずれも 1 人を相手にする
    手順（指示に従う／台本を流す／既存の映像を読む）なので、台数を増やす対象ではない。
    """
    if args.guided or args.scenario or args.video:
        return False
    source = config.get("source", {})
    if source.get("type", "webcam") not in ("remote", "iphone"):
        return False
    net = source.get("remote", source.get("iphone", {}))
    return int(net.get("max_peers", 1)) > 1
