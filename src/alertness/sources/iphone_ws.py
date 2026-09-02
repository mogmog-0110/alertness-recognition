"""iPhone からの映像入力。WebSocket サーバとして待ち受ける。

webcam と違い、こちらは待ち受け側で、端末が繋ぎに来る。判定結果を返すのに同じ接続が
要るので、接続は IPhoneLink が持ち、映像を読む source と結果を返す sink がそれを共有する。

asyncio は専用スレッドで回す。app.py は同期のループなので、そこへイベントループを
持ち込まない。

MJPEG 経路と違い、撮影時刻は端末が付けて送ってくる。受信時刻で代用してはいけない。
まばたき率・あくび間隔・rPPG は時間方向の量なので、通信が詰まった分だけ生理指標が歪む。
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import threading
import time
from collections.abc import Iterator

import cv2
import numpy as np

from ..contracts import Frame
from ._latest import LatestFrame

# 先頭 8 バイトが撮影時刻（秒, float64 リトルエンディアン）。残りが JPEG。
_HEADER = struct.Struct("<d")
_MAX_MESSAGE_BYTES = 8 * 1024 * 1024
_POLL_SECONDS = 0.001
# 時刻が戻ったフレームを一切通さないための最小の刻み。検出器は時刻をミリ秒の整数に
# 落とすので、これより細かく刻んでも同じ値になり「増えていない」と見なされる。
_MIN_STEP_SECONDS = 0.001


class IPhoneLink:
    """端末との WebSocket 接続。最新フレームを保持し、判定結果を送り返す。

    1 台だけを相手にする。2 台目が繋いできたら、結果の宛先は新しい方に移る。
    """

    def __init__(
        self,
        host: str = "0.0.0.0",  # noqa: S104 - 同じ Wi-Fi の端末から繋ぐので全 IF で待つ
        port: int = 8765,
        max_message_bytes: int = _MAX_MESSAGE_BYTES,
        certfile: str = "",
        keyfile: str = "",
        web_root: str = "",
    ) -> None:
        self._host = host
        self._requested_port = port
        self._max_message_bytes = max_message_bytes
        self._certfile = certfile
        self._keyfile = keyfile
        self._web_root = os.path.abspath(web_root) if web_root else None
        self._latest = LatestFrame()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._finished: asyncio.Event | None = None
        self._ws = None
        self._port = 0
        self._new_session = True
        self._offset = 0.0
        self._last_stamped = float("-inf")
        self._last_seen = 0.0
        self._ready = threading.Event()
        self._commands: list[str] = []
        self._commands_lock = threading.Lock()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        """実際に待ち受けている番号。port=0 を渡したときはここで確認する。"""
        return self._port

    @property
    def connected(self) -> bool:
        return self._ws is not None

    def wait_ready(self, timeout: float = 5.0) -> None:
        """待ち受けが始まるまで待つ。始められなければ理由を添えて投げる。

        番号が使用中なら、黙って繋がらないのではなくここで分かるようにする。
        「iPhone が繋がらない」という症状は原因が見えにくいので。
        """
        if not self._ready.wait(timeout):
            raise TimeoutError(f"WebSocket の待ち受けが {timeout:.0f} 秒で始まりませんでした。")
        if self._error is not None:
            raise RuntimeError(
                f"ポート {self._requested_port} で待ち受けられませんでした"
                f"（{type(self._error).__name__}）。"
                "他のプロセスが使っていないか、source.iphone.port を確認してください。"
            ) from self._error

    def take_newer_than(self, served: int):
        """served より新しい 1 枚。まだ無ければ None。"""
        return self._latest.take_newer_than(served)

    def send(self, payload: dict) -> None:
        """判定結果を端末へ返す。繋がっていなければ何もしない。"""
        ws, loop = self._ws, self._loop
        if ws is None or loop is None:
            return
        text = json.dumps(payload, ensure_ascii=False)
        try:
            asyncio.run_coroutine_threadsafe(ws.send(text), loop)
        except RuntimeError:
            # 送ろうとした瞬間にループが畳まれた。次のフレームで送り直せばよい。
            pass

    def close(self) -> None:
        loop, finished = self._loop, self._finished
        if loop is not None and finished is not None:
            try:
                loop.call_soon_threadsafe(finished.set)
            except RuntimeError:
                pass  # 待ち受けに失敗して既に畳まれている。閉じるものは無い
        self._thread.join(timeout=2.0)
        self._latest.clear()

    # ── 待ち受けスレッド ────────────────────────────────────
    def _serve(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as error:  # noqa: BLE001 - 起動失敗の理由を呼び出し側へ運ぶ
            self._error = error
        finally:
            self._ready.set()

    async def _main(self) -> None:
        from websockets.asyncio.server import serve

        self._loop = asyncio.get_running_loop()
        self._finished = asyncio.Event()
        context = self._ssl_context()
        async with serve(
            self._on_connect,
            self._host,
            self._requested_port,
            max_size=self._max_message_bytes,
            ssl=context,
            process_request=self._serve_page,
        ) as server:
            self._port = server.sockets[0].getsockname()[1]
            self._ready.set()
            scheme = "wss" if context else "ws"
            print(f"[iphone] 待ち受け {scheme}://{self._host}:{self._port}")
            if self._web_root:
                page = "https" if context else "http"
                print(f"[iphone] ブラウザ版: {page}://<このPCのIP>:{self._port}/")
            await self._finished.wait()

    def _ssl_context(self):
        """証明書があれば TLS で待ち受ける。

        iOS Safari は HTTPS でないとカメラを許可しない。ページと WebSocket を
        同じポートで出すので、端末側で承認する証明書は 1 つで済む。
        """
        if not (self._certfile and self._keyfile):
            return None
        import ssl

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self._certfile, self._keyfile)
        return context

    def _serve_page(self, connection, request):
        """WebSocket でないリクエストにはページを返す。

        別ポートで配ると証明書の承認が 2 回要るうえ、片方だけ承認して繋がらない
        という分かりにくい失敗をする。同じポートに寄せる。
        """
        if self._web_root is None:
            return None
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None  # WebSocket はそのまま通す

        import os
        from http import HTTPStatus

        name = request.path.split("?", 1)[0].lstrip("/") or "index.html"
        # ディレクトリを抜けられないようにする。
        path = os.path.normpath(os.path.join(self._web_root, name))
        if not path.startswith(os.path.abspath(self._web_root)) or not os.path.isfile(path):
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")

        from websockets.datastructures import Headers
        from websockets.http11 import Response

        with open(path, "rb") as handle:
            body = handle.read()
        content_type = (
            "text/html; charset=utf-8" if path.endswith(".html") else "application/octet-stream"
        )
        headers = Headers()
        headers["Content-Type"] = content_type
        headers["Content-Length"] = str(len(body))
        headers["Cache-Control"] = "no-store"
        return Response(HTTPStatus.OK.value, HTTPStatus.OK.phrase, headers, body)

    async def _on_connect(self, ws) -> None:
        from websockets.exceptions import ConnectionClosed

        self._ws = ws
        self._new_session = True
        print("[iphone] 接続しました。")
        try:
            async for message in ws:
                self._accept(message)
        except ConnectionClosed:
            pass  # 通信が切れるのは車内では珍しくない。待ち受けは続ける
        finally:
            if self._ws is ws:
                self._ws = None
            print("[iphone] 接続が切れました。待ち受けを続けます。")

    def _accept(self, message) -> None:
        """1 メッセージを取り込む。壊れていれば黙って捨てる。

        バイナリは映像、テキストは制御命令。取りこぼしは正常な動作（端末は送信が
        詰まると新しいフレームを捨てる）なので、1 枚読めなかったことを異常として
        扱わない。流れが止まったこと自体は Watchdog が知らせる。
        """
        if isinstance(message, str):
            self._accept_command(message)
            return
        if not isinstance(message, bytes | bytearray) or len(message) <= _HEADER.size:
            return
        (captured,) = _HEADER.unpack_from(message, 0)
        payload = np.frombuffer(memoryview(message)[_HEADER.size :], dtype=np.uint8)
        image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
        if image is None:
            return
        # 常に最新の 1 枚だけ。遅れて届いた過去のフレームに価値は無い。
        self._latest.put(image, self._stamp(float(captured)))

    def _accept_command(self, text: str) -> None:
        """端末からの制御命令。いまは再キャリブだけ。

        端末は運転者の手元にあり、PC の画面もキーボードも触れない。基準を取り直す
        手段が PC 側のキー操作しか無いと、実験のたびにプロセスを落とすことになる。
        """
        try:
            command = json.loads(text).get("command", "")
        except (ValueError, AttributeError):
            return
        if not isinstance(command, str) or not command:
            return
        with self._commands_lock:
            self._commands.append(command)

    def take_commands(self) -> list[str]:
        """溜まった命令を取り出す。読んだ分は消える。"""
        with self._commands_lock:
            pending, self._commands = self._commands, []
        return pending

    def _stamp(self, captured: float) -> float:
        """撮影時刻を、前へ戻らない時刻に直す。

        区間（1 回の接続）の中では端末の値をそのまま使う。まばたき率・あくび間隔・rPPG
        は間隔の量なので、ここを触ると生理指標が丸ごとずれる。触るのは区間の境目だけで、
        繋ぎ直しで 0 付近へ戻った分を、切れていた実時間ぶん前へずらして繋ぐ。

        戻る時刻を通すと検出器（MediaPipe の detect_for_video）が例外を投げ、そこから
        先のフレームも全部同じ理由で落ちる。1 回の切断で判定が二度と戻らなくなる。
        """
        now = time.monotonic()
        if self._new_session:
            self._new_session = False
            if self._last_stamped > float("-inf"):
                self._offset = self._last_stamped + (now - self._last_seen) - captured
        stamped = max(captured + self._offset, self._last_stamped + _MIN_STEP_SECONDS)
        self._last_stamped = stamped
        self._last_seen = now
        return stamped


class IPhoneSource:
    """FrameSource。IPhoneLink が受けた最新フレームを流す。

    繋がっていない間は何も yield しない。app.py のループはそこで止まるので、
    画面もキーも無い運転で終われるように interrupt() を持つ。
    """

    def __init__(self, link: IPhoneLink) -> None:
        self._link = link
        self._index = 0
        self._stop = threading.Event()

    @property
    def link(self) -> IPhoneLink:
        """結果の返送に使う接続。sink がここから同じ接続を掴む。"""
        return self._link

    def take_commands(self) -> list[str]:
        """端末から届いた制御命令。読んだ分は消える。"""
        return self._link.take_commands()

    def frames(self) -> Iterator[Frame]:
        served = 0
        while not self._stop.is_set():
            latest = self._link.take_newer_than(served)
            if latest is None:
                time.sleep(_POLL_SECONDS)
                continue
            served, image, captured = latest
            yield Frame(image=image, index=self._index, timestamp=captured, source_id="iphone")
            self._index += 1

    def interrupt(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self._stop.set()
        self._link.close()
