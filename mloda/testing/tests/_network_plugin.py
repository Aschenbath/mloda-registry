"""Synthetic hook and partial-request probes run in disposable subprocesses."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
from collections.abc import Mapping
from contextlib import ExitStack, closing
from http.client import HTTPConnection
from typing import Any
from unittest.mock import patch
from urllib.parse import urlsplit

TELEMETRY_BODY = b"telemetry from explicit config"
SINK_BODY = b"synthetic plugin result"


def _post(endpoint: str, body: bytes) -> None:
    parsed = urlsplit(endpoint)
    assert parsed.scheme == "http" and parsed.hostname == "127.0.0.1"
    assert parsed.port is not None
    with closing(HTTPConnection(parsed.hostname, parsed.port, timeout=3)) as connection:
        connection.request("POST", parsed.path or "/", body=body)
        with connection.getresponse() as response:
            response.read()
            assert 200 <= response.status < 300


def run_hook(env: Mapping[str, str]) -> dict[str, Any]:
    if env.get("DO_NOT_TRACK") != "1":
        _post(env["PLUGIN_ENDPOINT"], TELEMETRY_BODY)
    _post(env["PLUGIN_ENDPOINT"], SINK_BODY)
    return {
        "configured_do_not_track": env.get("DO_NOT_TRACK"),
        "configured_proxy_keys": sorted(key for key in env if key.lower().endswith("_proxy")),
        "sentinel": env["PLUGIN_SENTINEL"],
    }


def probe_partial_request(part: str) -> dict[str, Any]:
    from mloda.testing.network import recording_http_server

    fragments = {
        "header": b"POST / HTTP/1.1\r\nHost: 127.0.0.1\r\n",
        "body": b"POST / HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 12\r\n\r\nshort",
    }
    accepted = threading.Event()
    original_accept = socket.socket.accept

    def track_accept(listener: socket.socket) -> tuple[socket.socket, Any]:
        connection = original_accept(listener)
        accepted.set()
        return connection

    before = set(threading.enumerate())
    with ExitStack() as clients, patch.object(socket.socket, "accept", track_accept):
        with recording_http_server() as server:
            parsed = urlsplit(server.url)
            assert parsed.scheme == "http" and parsed.hostname == "127.0.0.1"
            assert parsed.port is not None
            client = clients.enter_context(socket.create_connection((parsed.hostname, parsed.port), timeout=3))
            client.sendall(fragments[part])
            assert accepted.wait(3), "server did not accept the incomplete request"

        return {
            "accepted": accepted.is_set(),
            "body_count": len(server.bodies),
            "client_open": client.fileno() >= 0,
            "threads_stopped": not (set(threading.enumerate()) - before),
        }


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--partial-request":
        report = probe_partial_request(sys.argv[2])
    else:
        report = run_hook(json.load(sys.stdin))
        report["process_do_not_track"] = os.environ.get("DO_NOT_TRACK")
        report["process_proxy_keys"] = sorted(key for key in os.environ if key.lower().endswith("_proxy"))
    print(json.dumps(report))
