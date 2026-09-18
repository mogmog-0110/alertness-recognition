"""複数端末を同時に判定する経路。

単独運転の App は「1 つの source・1 つの pipeline」を前提にしている。判定は
人ごとの状態（キャリブ基準・時間方向の履歴）を持つので、台数ぶんの pipeline を
台数ぶん独立に回す必要があり、1 つの App には収まらない。

RemoteHub が新しい端末を受けるたびに DeviceWorker を 1 つ立て、専用スレッドで
1 台ぶんの「観測→分類→送信」を回す。GUI・キー操作・ガイド収録・シナリオ再生は
単独運転専用の機能なので、ここでは持たない（運転者は端末側の画面だけを見る）。

端末が繋ぐたびに pipeline を作るのは重い処理（MediaPipe の初期化）なので、
待ち受けスレッドをそこで塞がないよう、DeviceWorker の生成と実行を別スレッドに
逃がす。塞ぐと、複数人がほぼ同時に繋いだときに後続の接続がそこで詰まる。
"""

from __future__ import annotations

import signal
import threading
import time
from typing import Any

from . import factory, log
from .app import _MAX_DETECT_FAILURES
from .sources.remote import RemoteSource
from .sources.remote_hub import Peer, RemoteHub
from .watchdog import Watchdog


class DeviceWorker:
    """1 台ぶんの判定ループ。App と違い GUI もキーも持たない。"""

    def __init__(self, peer: Peer, config: dict[str, Any]) -> None:
        from .feedback.remote import RemoteSink

        # Peer は RemoteLink と同じ形（connected/take_newer_than/take_commands/close）
        # を持つが、RemoteSource の型注釈は RemoteLink 固定なので構造的には合わない。
        self.source = RemoteSource(peer)  # type: ignore[arg-type]
        self._peer = peer
        self._config = config
        self._pipeline = factory.build_pipeline(config)
        self._calibrator = factory.build_calibrator(config)
        feedback = config.get("feedback", {})
        self._sink = RemoteSink(peer, **factory.remote_sink_options(feedback))
        self._calibrating = config.get("calibration", {}).get("enabled", True)
        self._detect_failures = 0
        self._peer_id = peer.id
        # App._on_stall/_on_recover と同じ考え方: 繋がっていない間の沈黙は異常では
        # ないので黙り、実際に知らせた分だけ「復帰」も言う。
        self._stall_reported = False
        self._watchdog = Watchdog(
            stall_seconds=feedback.get("stall_seconds", 3.0),
            repeat_seconds=feedback.get("stall_repeat_seconds", 5.0),
            on_stall=self._on_stall,
            on_recover=self._on_recover,
        )
        # 台数ぶん同時に走るので、基準の保存先は共有しない（同じファイルを取り合う）。
        # 単独運転の --record 相当（CSV 保存）も、被験者IDが台ごとに要るため未対応。

    def run(self) -> None:
        try:
            for frame in self.source.frames():
                # 最初の 1 枚が来てから見張る。端末が繋ぐ前後の沈黙は異常ではない。
                self._watchdog.start()
                self._watchdog.beat()
                for command in self.source.take_commands():
                    self._handle_command(command)
                obs = self._observe(frame)
                if obs is None:
                    continue
                if self._calibrating:
                    self._calibrator.collect(obs)
                    self._sink.calibrating(obs, self._calibrator.progress)
                    if self._calibrator.progress >= 1.0:
                        self._pipeline.set_profile(self._calibrator.finalize())
                        self._calibrating = False
                else:
                    assessment = self._pipeline.classify(obs)
                    self._sink.emit(obs, assessment)
        finally:
            self._watchdog.close()
            self._pipeline.close()
            self.source.close()

    def _observe(self, frame: Any) -> Any:
        # App._observe と同じ考え方: 単発の失敗はこのフレームだけ捨てて続ける。
        try:
            obs = self._pipeline.observe(frame)
        except Exception as error:  # noqa: BLE001 - 検出器側の例外型は実装依存
            self._detect_failures += 1
            if self._detect_failures >= _MAX_DETECT_FAILURES:
                print(
                    f"[remote] {self._peer_id} 台目の検出器が壊れました（{type(error).__name__}）。"
                )
                raise
            return None
        self._detect_failures = 0
        return obs

    def _handle_command(self, command: str) -> None:
        if command == "recalibrate":
            log.detail(f"[remote] {self._peer_id} 台目から再キャリブを受け取りました。")
            self._recalibrate()
        else:
            log.detail(f"[remote] {self._peer_id} 台目から未知の命令を無視しました: {command}")

    def _recalibrate(self) -> None:
        # 別人に替わった可能性があるので、本人前提で育てた基準と履歴も捨てる。
        # calibrator は StatisticalCalibrator が完了後も内部状態を持ち続けるので、
        # 使い回すと progress が即座に 1.0 のままになる。ここで作り直す。
        self._pipeline.reset_state()
        self._calibrator = factory.build_calibrator(self._config)
        self._calibrating = True

    def _on_stall(self, silent: float) -> None:
        # 端末が繋がっていないのは待ち受け中の正常な姿で、操作者が直すものではない。
        if not self._peer.connected:
            return
        self._stall_reported = True
        # 端末の画面が自分で止まった/切れた表示に変わる（黄色表示）ので、PC 側の
        # コンソールには要らない。台数が増えるほど、繋ぎ直しのたびに埋もれてしまう。
        log.detail(f"[remote] {self._peer_id} 台目の判定が {silent:.1f} 秒とまっています。")

    def _on_recover(self, _silent: float) -> None:
        if not self._stall_reported:
            return
        self._stall_reported = False
        log.detail(f"[remote] {self._peer_id} 台目の判定が再開しました。")


class MultiDeviceApp:
    """RemoteHub の待ち受けと、端末ごとの DeviceWorker の起動・後始末を束ねる。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        source = config.get("source", {})
        net = source.get("remote", source.get("iphone", {}))
        self._max_peers = max(1, int(net.get("max_peers", 1)))
        self._workers: dict[int, DeviceWorker] = {}
        self._threads: dict[int, threading.Thread] = {}
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._hub = RemoteHub(
            net.get("host", "0.0.0.0"),
            int(net.get("port", 8765)),
            max_peers=self._max_peers,
            certfile=net.get("certfile", ""),
            keyfile=net.get("keyfile", ""),
            web_root=net.get("web_root", ""),
            advertise_host=net.get("advertise_host", ""),
            on_connect=self._on_connect,
            on_disconnect=self._on_disconnect,
        )
        self._hub.wait_ready()

    def run(self) -> int:
        self._install_signal_handlers()
        from .feedback.connect_qr import ConnectQr

        prompt = None
        if self._hub.url and self._config.get("feedback", {}).get("connect_qr", True):
            prompt = ConnectQr(self._hub.url)
        try:
            while not self._stopping.is_set():
                if prompt is not None:
                    if self._hub.connected_count < self._max_peers:
                        if not prompt.waiting():
                            break  # QR の窓で 'q'
                    else:
                        prompt.hide()
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            if prompt is not None:
                prompt.hide()
            self._close()
        return 0

    def _on_connect(self, peer: Peer) -> None:
        # pipeline の生成（MediaPipe の初期化）は重いので、待ち受けスレッドを
        # 塞がないよう別スレッドへ逃がす。ここで待つと後続の接続が詰まる。
        thread = threading.Thread(
            target=self._run_worker, args=(peer,), daemon=True, name=f"device-{peer.id}"
        )
        with self._lock:
            self._threads[peer.id] = thread
        thread.start()

    def _run_worker(self, peer: Peer) -> None:
        worker = DeviceWorker(peer, self._config)
        with self._lock:
            self._workers[peer.id] = worker
        try:
            worker.run()
        finally:
            with self._lock:
                self._workers.pop(peer.id, None)
                self._threads.pop(peer.id, None)

    def _on_disconnect(self, peer: Peer) -> None:
        with self._lock:
            worker = self._workers.get(peer.id)
        if worker is not None:
            worker.source.interrupt()

    def _install_signal_handlers(self) -> None:
        def handler(_signum, _frame):
            self._stopping.set()

        for name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                continue

    def _close(self) -> None:
        self._hub.close()  # 全接続を閉じる。各 Peer の切断で worker も自然に終わる
        with self._lock:
            threads = list(self._threads.values())
        for thread in threads:
            thread.join(timeout=3.0)


def run(config: dict[str, Any]) -> int:
    return MultiDeviceApp(config).run()
