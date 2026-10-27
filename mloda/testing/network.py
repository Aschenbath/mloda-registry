"""Loopback HTTP recording and explicit environments for plugin subprocess tests."""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread


class _RecordingServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self) -> None:
        self.bodies: list[bytes] = []
        self._connections: set[socket.socket] = set()
        self._connections_lock = Lock()
        super().__init__(("127.0.0.1", 0), _RecordingHandler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    def get_request(self) -> tuple[socket.socket, tuple[str, int]]:
        connection, address = super().get_request()
        with self._connections_lock:
            self._connections.add(connection)
        return connection, address

    def shutdown_request(self, request: socket.socket | tuple[bytes, socket.socket]) -> None:
        try:
            super().shutdown_request(request)
        finally:
            if isinstance(request, socket.socket):
                with self._connections_lock:
                    self._connections.discard(request)

    def close_active_connections(self) -> None:
        with self._connections_lock:
            connections = tuple(self._connections)
        for connection in connections:
            with suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)


class _RecordingHandler(BaseHTTPRequestHandler):
    server: _RecordingServer
    # On Windows, shutdown alone may not wake a stalled socket read.
    timeout = 1.0

    def handle(self) -> None:
        # Context teardown disconnects clients that have not finished their request.
        with suppress(OSError):
            super().handle()

    def do_POST(self) -> None:
        if "Transfer-Encoding" in self.headers:
            self.send_error(HTTPStatus.NOT_IMPLEMENTED, "Transfer-Encoding is not supported")
            return

        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1:
            self.send_error(HTTPStatus.BAD_REQUEST, "Exactly one Content-Length header is required")
            return
        length = lengths[0].strip(" \t")
        if not length.isascii() or not length.isdecimal():
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
            return
        try:
            remaining = int(length)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
            return

        body = bytearray()
        while remaining:
            chunk = self.rfile.read(min(remaining, 64 * 1024))
            if not chunk:
                self.send_error(HTTPStatus.BAD_REQUEST, "Incomplete request body")
                return
            body.extend(chunk)
            remaining -= len(chunk)
        self.server.bodies.append(bytes(body))

        response = b"{}"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        # Keep routine HTTP request logs out of the test process's stderr.
        pass


@contextmanager
def recording_http_server() -> Iterator[_RecordingServer]:
    """Yield a loopback-only server with ``url`` and ordered raw POST ``bodies``.

    POSTs need one nonnegative Content-Length and no Transfer-Encoding. Successful
    requests receive HTTP 200 with an empty JSON object. Socket I/O has a one-second
    timeout so stalled headers or bodies cannot prevent thread cleanup on Windows.
    On exit, incomplete requests are disconnected and all server threads finish;
    bodies remain readable.
    """
    with _RecordingServer() as server:
        thread = Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
            name="mloda-recording-http-server",
            daemon=False,
        )
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            server.close_active_connections()
            thread.join()


def subprocess_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy an environment, remove case-insensitive *_proxy keys and opt out of tracking.

    Only ``None`` inherits os.environ; an explicit empty mapping stays empty apart
    from DO_NOT_TRACK=1. All casing variants are replaced with this uppercase key
    to avoid conflicting values in Windows subprocesses. Apply this to explicit
    plugin configuration mappings too. This preset only affects clients that honor
    it; it is not a network sandbox.
    """
    source = os.environ if env is None else env
    result = {
        key: value
        for key, value in source.items()
        if key.upper() != "DO_NOT_TRACK" and not key.lower().endswith("_proxy")
    }
    result["DO_NOT_TRACK"] = "1"
    return result
