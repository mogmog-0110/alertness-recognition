"""複数端末を同時に受け付ける待ち受け（複数台対応）。

RemoteLink は 1 接続を「枠」として持ち、2 台目が繋いできたら古い方を閉じて
入れ替える（同じ端末が繋ぎ直した場合を含め、常に「今つながっている 1 人」だけを
判定する設計）。複数台では逆に、同時に繋いだ全員を判定し続けたい。

Peer は「1 回の接続」を寿命とする単純なモデルにする。繋ぎ直しは新しい Peer として
扱い、判定状態（キャリブレーション等）もそこで作り直す。ページの開き直しを見分けて
基準を捨て直す RemoteLink の hello/NEW_SESSION の仕組みは、ここでは要らない
（接続そのものが変わればどのみち新しい Peer になり、判定状態も最初から作られる）。

max_peers を超える接続は待ち行列に並べず、即座に閉じる。順番待ちをさせると
「繋がらない」のと区別がつかない失敗になるので、繋げないなら繋げないとすぐ返す。

RemoteSource が link に期待する形（connected/take_newer_than/take_commands/close）
は RemoteLink と同じにしてあるので、RemoteSource(peer) がそのまま使える。
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import threading
from collections.abc import Callable

from .. import log
from ._latest import LatestFrame

_HEADER = struct.Struct("<d")
_MAX_MESSAGE_BYTES = 8 * 1024 * 1024
_MIN_STEP_SECONDS = 0.001
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
}

OnPeerEvent = Callable[["Peer"], None]


class Peer:
    """1 端末との 1 回の接続。切れたら寿命も終わる（繋ぎ直しは新しい Peer になる）。"""

    def __init__(self, peer_id: int) -> None:
        self.id = peer_id
        self._ws = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._latest: LatestFrame[memoryview] = LatestFrame()
        self._commands: list[str] = []
        self._commands_lock = threading.Lock()
        self._last_stamped = float("-inf")
        self._closed = threading.Event()

    @property
    def connected(self) -> bool:
        return not self._closed.is_set()

    def take_newer_than(self, served: int):
        return self._latest.take_newer_than(served)

    def send(self, payload: dict) -> None:
        ws, loop = self._ws, self._loop
        if ws is None or loop is None:
            return
        text = json.dumps(payload, ensure_ascii=False)
        try:
            asyncio.run_coroutine_threadsafe(ws.send(text), loop)
        except RuntimeError:
            pass  # 送ろうとした瞬間に切れた。次のフレームで送り直せばよい

    def take_commands(self) -> list[str]:
        with self._commands_lock:
            pending, self._commands = self._commands, []
        return pending

    def close(self) -> None:
        """この接続だけ切る。ハブ全体（待ち受けそのもの）は畳まない。"""
        ws, loop = self._ws, self._loop
        self._closed.set()
        if ws is not None and loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(ws.close(), loop)
            except RuntimeError:
                pass

    def _accept(self, message) -> dict[str, str] | None:
        """1 メッセージを取り込む。RemoteLink._accept と同じ形式・同じ意味。"""
        if isinstance(message, str):
            return self._accept_command(message)
        if not isinstance(message, bytes | bytearray) or len(message) <= _HEADER.size:
            return None
        (captured,) = _HEADER.unpack_from(message, 0)
        # 復号は取り出す側（RemoteSource）で行う。判定が追いつかない間に届いた分は
        # 上書きされて捨てられるので、ここで復号すると無駄になる。
        stamped = max(captured, self._last_stamped + _MIN_STEP_SECONDS)
        self._last_stamped = stamped
        self._latest.put(memoryview(message)[_HEADER.size :], stamped)
        return None

    def _accept_command(self, text: str) -> dict[str, str] | None:
        try:
            payload = json.loads(text)
        except (ValueError, AttributeError):
            return None
        if not isinstance(payload, dict):
            return None
        kind = payload.get("type")
        if kind == "ping":
            return {"type": "pong"}
        if kind == "hello":
            # 接続そのものが Peer の寿命と一致するので、開き直しを見分ける必要はない。
            return None
        command = payload.get("command", "")
        if not isinstance(command, str) or not command:
            return None
        with self._commands_lock:
            self._commands.append(command)
        return None


class RemoteHub:
    """1 つのポートで、最大 max_peers 台までを同時に受け付ける。

    新しい端末が繋ぐたびに on_connect(peer) を呼ぶので、呼び出し側は端末ごとに
    独立した判定状態（pipeline・calibrator）をその場で作れる。切れたときは
    on_disconnect(peer) で知らせる。
    """

    def __init__(
        self,
        host: str = "0.0.0.0",  # noqa: S104 - 同じ Wi-Fi の端末から繋ぐので全 IF で待つ
        port: int = 8765,
        max_peers: int = 3,
        max_message_bytes: int = _MAX_MESSAGE_BYTES,
        certfile: str = "",
        keyfile: str = "",
        web_root: str = "",
        advertise_host: str = "",
        on_connect: OnPeerEvent | None = None,
        on_disconnect: OnPeerEvent | None = None,
    ) -> None:
        self._host = host
        self._requested_port = port
        self._max_peers = max_peers
        self._max_message_bytes = max_message_bytes
        self._certfile = certfile
        self._keyfile = keyfile
        self._web_root = os.path.abspath(web_root) if web_root else None
        self._advertise_host = advertise_host
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._address = ""
        self._peers: list[Peer] = []
        self._peers_lock = threading.Lock()
        self._next_id = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._finished: asyncio.Event | None = None
        self._port = 0
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return self._port

    @property
    def connected_count(self) -> int:
        with self._peers_lock:
            return len(self._peers)

    @property
    def url(self) -> str:
        """端末のブラウザで開く URL。wait_ready() の後で読む。空なら案内できない。"""
        if self._web_root is None or not self._address:
            return ""
        return f"https://{self._address}:{self._port}/"

    def wait_ready(self, timeout: float = 5.0) -> None:
        if not self._ready.wait(timeout):
            raise TimeoutError(f"WebSocket の待ち受けが {timeout:.0f} 秒で始まりませんでした。")
        if self._error is not None:
            raise RuntimeError(
                f"ポート {self._requested_port} で待ち受けられませんでした"
                f"（{type(self._error).__name__}: {self._error}）。"
                "他のプロセスが使っていないか、source.remote.port を確認してください。"
            ) from self._error

    def close(self) -> None:
        loop, finished = self._loop, self._finished
        if loop is not None and finished is not None:
            try:
                loop.call_soon_threadsafe(finished.set)
            except RuntimeError:
                pass
        self._thread.join(timeout=2.0)
        with self._peers_lock:
            peers, self._peers = self._peers, []
        for peer in peers:
            peer._latest.clear()

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
            self._on_ws_connect,
            self._host,
            self._requested_port,
            max_size=self._max_message_bytes,
            ssl=context,
            process_request=self._serve_page,
            ping_interval=5,
            ping_timeout=10,
            close_timeout=1,
        ) as server:
            self._port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._finished.wait()

    def _ssl_context(self):
        if not (self._certfile and self._keyfile):
            return None
        import ssl

        from ..webcert import ensure

        self._address, renewed = ensure(self._certfile, self._keyfile, self._advertise_host)
        if renewed:
            print(f"[remote] {self._address} 用の証明書を作りました（端末で再度承認が要ります）")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self._certfile, self._keyfile)
        return context

    def _serve_page(self, connection, request):
        if self._web_root is None:
            return None
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None  # WebSocket はそのまま通す

        from http import HTTPStatus

        name = request.path.split("?", 1)[0].lstrip("/") or "index.html"
        path = os.path.normpath(os.path.join(self._web_root, name))
        if not path.startswith(self._web_root + os.sep) or not os.path.isfile(path):
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")

        from websockets.datastructures import Headers
        from websockets.http11 import Response

        with open(path, "rb") as handle:
            body = handle.read()
        extension = os.path.splitext(path)[1].lower()
        headers = Headers()
        headers["Content-Type"] = _CONTENT_TYPES.get(extension, "application/octet-stream")
        headers["Content-Length"] = str(len(body))
        headers["Cache-Control"] = "no-store"
        return Response(HTTPStatus.OK.value, HTTPStatus.OK.phrase, headers, body)

    async def _on_ws_connect(self, ws) -> None:
        from websockets.exceptions import ConnectionClosed

        peer = self._register(ws)
        if peer is None:
            # 満員。順番待ちにはしない（繋がらないのと区別がつかなくなる）。
            print(f"[remote] {self._max_peers} 台で満員のため、新しい接続を断りました。")
            await ws.close(code=1013, reason="too many devices")
            return
        log.detail(
            f"[remote] {peer.id} 台目が接続しました（{self.connected_count}/{self._max_peers}）"
        )
        if self._on_connect is not None:
            self._on_connect(peer)
        try:
            async for message in ws:
                reply = peer._accept(message)
                if reply is not None:
                    await ws.send(json.dumps(reply, ensure_ascii=False))
        except ConnectionClosed:
            pass  # 通信が切れるのは珍しくない。この Peer だけ畳んで待ち受けは続ける
        finally:
            self._unregister(peer)
            log.detail(f"[remote] {peer.id} 台目の接続が切れました。")
            if self._on_disconnect is not None:
                self._on_disconnect(peer)

    def _register(self, ws) -> Peer | None:
        with self._peers_lock:
            if len(self._peers) >= self._max_peers:
                return None
            self._next_id += 1
            peer = Peer(self._next_id)
            self._peers.append(peer)
        peer._ws = ws
        peer._loop = asyncio.get_running_loop()
        return peer

    def _unregister(self, peer: Peer) -> None:
        peer.close()
        with self._peers_lock:
            if peer in self._peers:
                self._peers.remove(peer)
