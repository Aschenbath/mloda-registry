"""Loopback service stand-ins and explicit subprocess environments stay isolated."""

from __future__ import annotations

import json
import os
import socket
import subprocess  # nosec B404
import sys
import threading
from contextlib import closing, nullcontext
from http.client import HTTPConnection
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

import pytest

from mloda.testing.tests._network_plugin import SINK_BODY, TELEMETRY_BODY

_PROXIES = {
    "HTTP_PROXY": "http://127.0.0.1:9",
    "http_proxy": "http://127.0.0.1:9",
    "HtTpS_pRoXy": "http://127.0.0.1:9",
    "ALL_PROXY": "http://127.0.0.1:9",
    "ftp_proxy": "http://127.0.0.1:9",
    "NO_PROXY": "*",
    "custom_PrOxY": "http://127.0.0.1:9",
}


def _address(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    assert parsed.scheme == "http"
    assert parsed.hostname == "127.0.0.1"
    assert parsed.port is not None and 0 < parsed.port < 65536
    return parsed.hostname, parsed.port


def _request(
    url: str,
    body: bytes,
    *,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    chunked: bool = False,
    check_json: bool = False,
) -> int:
    host, port = _address(url)
    with closing(HTTPConnection(host, port, timeout=3)) as connection:
        connection.request(method, "/", body=body, headers=headers or {}, encode_chunked=chunked)
        with connection.getresponse() as response:
            response_body = response.read()
            if check_json:
                assert response.status == 200
                assert response.headers.get_content_type() == "application/json"
                content_length = response.getheader("Content-Length")
                assert content_length is not None
                assert int(content_length) == len(response_body)
                assert json.loads(response_body) == {}
            return response.status


def _run_hook(config: dict[str, str], env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(  # nosec B603
        [sys.executable, str(Path(__file__).with_name("_network_plugin.py"))],
        input=json.dumps(config),
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    report: dict[str, Any] = json.loads(completed.stdout)
    return report


def test_server_binds_only_loopback_and_records_raw_post_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    from mloda.testing.network import recording_http_server

    bindings: list[tuple[str, int]] = []
    original_bind = socket.socket.bind

    def track_bind(connection: socket.socket, address: Any) -> None:
        bindings.append(address)
        original_bind(connection, address)

    monkeypatch.setattr(socket.socket, "bind", track_bind)
    bodies = [b'{"event":"synthetic"}', b"\x00\xffbinary\r\n", b""]
    with recording_http_server() as server:
        _address(server.url)
        assert server.bodies == []
        for body in bodies:
            assert _request(server.url, body, check_json=True) == 200
        assert server.bodies == bodies

    assert server.bodies == bodies
    assert bindings
    assert all(host == "127.0.0.1" for host, _ in bindings)


def test_simultaneous_servers_have_separate_ports_and_recordings() -> None:
    from mloda.testing.network import recording_http_server

    with recording_http_server() as first, recording_http_server() as second:
        assert _address(first.url) != _address(second.url)
        assert 200 <= _request(first.url, b"first server") < 300
        assert 200 <= _request(second.url, b"second server") < 300
        assert first.bodies == [b"first server"]
        assert second.bodies == [b"second server"]


@pytest.mark.parametrize("exceptional", [False, True], ids=["normal", "exception"])
def test_context_exit_releases_listener_and_threads(exceptional: bool) -> None:
    from mloda.testing.network import recording_http_server

    before = set(threading.enumerate())
    owned_threads: set[threading.Thread] = set()
    address = ("127.0.0.1", 0)
    with pytest.raises(RuntimeError, match="intentional test error") if exceptional else nullcontext():
        with recording_http_server() as server:
            address = _address(server.url)
            assert 200 <= _request(server.url, b"before exit") < 300
            owned_threads = set(threading.enumerate()) - before
            if exceptional:
                raise RuntimeError("intentional test error")

    assert all(not thread.is_alive() for thread in owned_threads)
    with socket.socket() as probe:
        probe.settimeout(1)
        assert probe.connect_ex(address) != 0


@pytest.mark.parametrize("part", ["header", "body"])
def test_context_exit_finishes_with_an_accepted_incomplete_request(part: str) -> None:
    from mloda.testing.network import subprocess_env

    completed = subprocess.run(  # nosec B603
        [sys.executable, str(Path(__file__).with_name("_network_plugin.py")), "--partial-request", part],
        env=subprocess_env(),
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )

    assert json.loads(completed.stdout) == {
        "accepted": True,
        "body_count": 0,
        "client_open": True,
        "threads_stopped": True,
    }


@pytest.mark.parametrize(
    ("method", "headers"),
    [
        ("GET", {"Content-Length": "3"}),
        ("PUT", {"Content-Length": "3"}),
        ("POST", {"Transfer-Encoding": "chunked"}),
        ("POST", {"Content-Length": "-1"}),
        ("POST", {"Content-Length": "not-a-length"}),
    ],
    ids=["get", "put", "chunked", "negative-length", "invalid-length"],
)
def test_unsupported_requests_are_rejected_without_recording(method: str, headers: dict[str, str]) -> None:
    from mloda.testing.network import recording_http_server

    with recording_http_server() as server:
        host, port = _address(server.url)
        with closing(HTTPConnection(host, port, timeout=3)) as connection:
            connection.putrequest(method, "/")
            for name, value in headers.items():
                connection.putheader(name, value)
            connection.endheaders()
            # Unsupported methods or framing must be rejected without waiting for a body.
            with connection.getresponse() as response:
                response.read()
                assert 400 <= response.status < 600
        assert server.bodies == []


@pytest.mark.parametrize("read_only", [False, True], ids=["dict", "read-only-mapping"])
def test_subprocess_env_copies_explicit_mapping_and_preserves_its_inputs(
    read_only: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mloda.testing.network import subprocess_env

    ambient = {"DO_NOT_TRACK": "1", "AMBIENT_ONLY": "not inherited"}
    monkeypatch.setattr(os, "environ", ambient)
    source = {
        **_PROXIES,
        "DO_NOT_TRACK": "0",
        # Windows folds child environment keys in insertion order, so put the aliases last.
        "do_not_track": "0",
        "Do_Not_Track": "0",
        "EXPLICIT_ONLY": "kept",
        "PROXY_FEATURE": "kept",
    }
    original = dict(source)
    result = subprocess_env(MappingProxyType(source) if read_only else source)

    assert result == {"DO_NOT_TRACK": "1", "EXPLICIT_ONLY": "kept", "PROXY_FEATURE": "kept"}
    result["EXPLICIT_ONLY"] = "changed copy"
    assert source == original
    assert ambient == {"DO_NOT_TRACK": "1", "AMBIENT_ONLY": "not inherited"}


def test_subprocess_env_empty_mapping_does_not_inherit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from mloda.testing.network import subprocess_env

    ambient = {"DO_NOT_TRACK": "0", "AMBIENT_ONLY": "not inherited"}
    monkeypatch.setattr(os, "environ", ambient)
    source: dict[str, str] = {}

    assert subprocess_env(source) == {"DO_NOT_TRACK": "1"}
    assert source == {}
    assert ambient == {"DO_NOT_TRACK": "0", "AMBIENT_ONLY": "not inherited"}


def test_subprocess_env_defaults_to_a_copy_of_the_current_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from mloda.testing.network import subprocess_env

    ambient = {**_PROXIES, "DO_NOT_TRACK": "0", "PATH": "/test/bin", "SystemRoot": "test-root", "KEEP": "value"}
    original = dict(ambient)
    monkeypatch.setattr(os, "environ", ambient)

    result = subprocess_env()

    assert result == {"DO_NOT_TRACK": "1", "PATH": "/test/bin", "SystemRoot": "test-root", "KEEP": "value"}
    result["KEEP"] = "changed copy"
    assert ambient == original


def test_real_hook_subprocess_uses_the_sanitized_explicit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from mloda.testing.network import recording_http_server, subprocess_env

    monkeypatch.setenv("DO_NOT_TRACK", "1")
    guarded_env = {key: value for key, value in os.environ.items() if key.upper() in {"SYSTEMROOT", "PATH"}}
    guarded_env["DO_NOT_TRACK"] = "1"
    with recording_http_server() as server:
        configured = {
            **guarded_env,
            **_PROXIES,
            "DO_NOT_TRACK": "0",
            "do_not_track": "0",
            "Do_Not_Track": "0",
            "PLUGIN_ENDPOINT": server.url,
            "PLUGIN_SENTINEL": "explicit config",
        }
        original = dict(configured)
        bypassed = _run_hook(configured, guarded_env)
        prepared = subprocess_env(configured)
        isolated = _run_hook(prepared, prepared)

        assert bypassed["process_do_not_track"] == "1"
        assert bypassed["configured_do_not_track"] == "0"
        assert set(bypassed["configured_proxy_keys"]) == set(_PROXIES)
        assert isolated == {
            "configured_do_not_track": "1",
            "configured_proxy_keys": [],
            "sentinel": "explicit config",
            "process_do_not_track": "1",
            "process_proxy_keys": [],
        }
        assert server.bodies == [TELEMETRY_BODY, SINK_BODY, SINK_BODY]
        assert configured == original
    assert os.environ["DO_NOT_TRACK"] == "1"
