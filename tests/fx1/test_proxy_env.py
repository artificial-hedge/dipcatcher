"""Proxy variables copied into a child keep scheme, host, and port only."""

from __future__ import annotations

import json
import os
import sys

from fx1.interactive.procs import ProcessRegistry, ProcLimits, run_owned, scrub_env

_PASS = "proxy-secret-pass"
_OTHER = "s3cret-token"


def test_proxy_userinfo_never_reaches_the_child() -> None:
    env = scrub_env(
        {
            "PATH": os.environ.get("PATH", ""),
            "http_proxy": f"http://alice:{_PASS}@proxy.example:8080/hidden?x={_PASS}",
            "HTTPS_PROXY": f"https://bob:{_OTHER}@secure.example:443",
            "ALL_PROXY": "socks5://u:p@127.0.0.1:1080",
            "ftp_proxy": "ftp://u:p@files.example:21",
            "SOCKS_PROXY": "not a proxy",
            "NO_PROXY": "localhost",
        }
    )
    code = "import json, os, sys; json.dump(dict(os.environ), sys.stdout)"
    result = run_owned(
        [sys.executable, "-c", code],
        ProcLimits(timeout_s=30),
        env=env,
        registry=ProcessRegistry(),
    )
    assert result.exit_code == 0, result.stderr
    blob = result.stdout + result.stderr
    assert _PASS not in blob
    assert _OTHER not in blob
    assert "alice" not in blob
    payload = json.loads(result.stdout)
    assert payload["http_proxy"] == "http://proxy.example:8080"
    assert payload["HTTPS_PROXY"] == "https://secure.example:443"
    assert payload["ALL_PROXY"] == "socks5://127.0.0.1:1080"
    assert payload["ftp_proxy"] == "ftp://files.example:21"
    assert "SOCKS_PROXY" not in payload
    assert payload["NO_PROXY"] == "localhost"
