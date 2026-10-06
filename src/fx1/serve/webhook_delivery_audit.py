"""fx1 serve audit — webhook dispatcher internals (lane 184).

``webhook_audit`` pins the delivery contract at the route level; this
battery measures the dispatcher's own mechanics in
``src/fx1/serve/webhooks.py``: sign/verify byte-exactness and
tolerance boundaries, ``check_callback_url`` grammar, the SSRF pin
(literal hosts, resolved addresses, ipv4-mapped unwrap, private
opt-in env), the pinned-connection seam (TCP to the validated IP
while TLS authenticates the URL host), and ``deliver_signed``'s
retry classification, backoff schedule, per-attempt signature
freshness, address-walk order, and never-raise return contract —
against a scripted local HTTP server recording every delivery.

Loopback deliveries need ``FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS=1`` —
each leg sets it narrowly and restores the prior value in a
``finally``. Sleeps ride ``time.sleep`` patching, never wall-clock.
Measured bools only; a defect pins ``False`` and is fixed on this PR.
"""

from __future__ import annotations

import contextlib
import http.server
import os
import socket
import threading
from collections.abc import Iterator
from typing import Any
from unittest import mock

__all__ = ["webhook_delivery_audit", "webhook_delivery_audit_bench"]


# The exact battery contract: an audit that drops or renames a probe
# fails loudly here, and ``bench`` refuses to seal an incomplete or
# partial result set as a pass.
_EXPECTED_PROBES: frozenset[str] = frozenset(
    {
        "all_addresses_fault_retries_outer",
        "backoff_doubles",
        "backoff_honors_custom_value",
        "body_reaches_every_attempt",
        "conn_refused_retries_then_errors",
        "content_length_matches_body",
        "content_type_json",
        "definitive_4xx_no_retry",
        "duplicates_deduped",
        "each_attempt_verifies_on_raw_body",
        "empty_body_roundtrip",
        "empty_resolution_oserror",
        "env_allows_loopback_literal",
        "env_restored_after_leg",
        "env_yes_counts",
        "env_zero_does_not_count",
        "fail_fail_ok_delivers_third",
        "fault_walks_to_next_address",
        "five_xx_short_circuits_walk",
        "four_xx_ends_walk_definitively",
        "fresh_at_tolerance_edge_future",
        "fresh_at_tolerance_edge_past",
        "fresh_ts_per_attempt",
        "future_beyond_tolerance_refused",
        "garbage_refused",
        "hostname_not_validated_without_dns",
        "http_connect_dials_validated_ip",
        "http_connection_keeps_url_host",
        "http_ok",
        "https_connect_dials_validated_ip",
        "https_sni_is_url_host",
        "invalid_url_never_raises_0",
        "invalid_url_never_raises_1",
        "invalid_url_never_raises_2",
        "invalid_url_never_raises_3",
        "invalid_url_never_raises_4",
        "literal_private_refused_0",
        "literal_private_refused_1",
        "literal_private_refused_10",
        "literal_private_refused_11",
        "literal_private_refused_12",
        "literal_private_refused_13",
        "literal_private_refused_14",
        "literal_private_refused_15",
        "literal_private_refused_2",
        "literal_private_refused_3",
        "literal_private_refused_4",
        "literal_private_refused_5",
        "literal_private_refused_6",
        "literal_private_refused_7",
        "literal_private_refused_8",
        "literal_private_refused_9",
        "max_attempts_one_single_call",
        "max_attempts_zero_never_dials",
        "mixed_public_private_refused",
        "negative_tolerance_disables_freshness",
        "new_second_new_signature",
        "no_host_refused",
        "non_tcp_skipped_then_oserror",
        "none_passes_through",
        "nonfinite_now_refused",
        "nonfinite_tolerance_refused",
        "nonfinite_ts_refused",
        "path_and_query_reach_wire",
        "persistent_5xx_exhausts",
        "port_garbage_refused",
        "port_zero_refused",
        "private_allowed_under_env",
        "private_resolution_refused",
        "private_url_refused_zero_attempts",
        "public_passthrough",
        "public_v6_passthrough",
        "redirect_is_retried_not_followed",
        "resolution_fault_retries_and_errors",
        "resolved_17216_refused",
        "resolved_17232_public_passes",
        "resolved_cgnat_refused",
        "resolved_ula_refused",
        "root_target_is_slash",
        "same_second_same_signature",
        "scheme_file_refused",
        "scheme_ftp_refused",
        "scheme_ws_refused",
        "sign_format_sha256_hex",
        "signed_every_attempt",
        "signed_stale_still_refused",
        "slow_endpoint_times_out_retries",
        "stale_beyond_tolerance_refused",
        "status_204_delivered",
        "target_preserves_path_query",
        "unicode_body_roundtrip",
        "unsigned_sends_no_signature",
        "userinfo_pass_refused",
        "userinfo_refused",
        "v4_mapped_private_refused",
        "verify_never_raises_garbage",
        "verify_rejects_bare_hex",
        "verify_rejects_empty_secret",
        "verify_rejects_empty_sig",
        "verify_rejects_missing_sig",
        "verify_rejects_missing_ts",
        "verify_rejects_nonascii_sig",
        "verify_rejects_nonascii_ts",
        "verify_rejects_reserialized_body",
        "verify_rejects_tampered_body",
        "verify_rejects_tampered_ts",
        "verify_rejects_wrong_prefix",
        "verify_rejects_wrong_secret",
        "verify_roundtrip",
        "verify_uses_wallclock_default",
        "walk_fault_error_names_exception",
    }
)

_AUDIT_LOCK = threading.Lock()

_SECRET = "k3y-material-webhook"  # placeholder only — never a real token
_BODY = b'{"job_id":"j-1","status":"succeeded","result":{"ok":true}}'
_LOOPBACK = "127.0.0.1"  # NOSONAR(S1313) — scripted local server only
_ENV = "FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"


@contextlib.contextmanager
def _audit_context() -> Iterator[None]:
    """Serialize the battery and normalize ambient webhook env.

    A caller that legitimately exported
    ``FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS`` must not leak into the
    refusal probes — the value is saved, cleared for the run, and
    restored verbatim afterwards (present -> value, absent -> absent),
    even on fault paths.
    """
    _AUDIT_LOCK.acquire()
    prev = os.environ.get(_ENV)
    os.environ.pop(_ENV, None)
    try:
        yield
    finally:
        try:
            if prev is None:
                os.environ.pop(_ENV, None)
            else:
                os.environ[_ENV] = prev
        finally:
            _AUDIT_LOCK.release()


@contextlib.contextmanager
def _allow_private() -> Iterator[None]:
    """Scope the SSRF opt-in env to one leg and restore the prior
    value — matching the drain/webhook audit convention exactly."""
    prev = os.environ.get(_ENV)
    os.environ[_ENV] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = prev


# --------------------------------------------------------------------------
# scripted loopback server
# --------------------------------------------------------------------------


class _Recorder:
    """Threaded HTTP server replaying a status queue and recording
    every delivery's headers and raw body."""

    def __init__(self, statuses: list[int], delay_s: float = 0.0) -> None:
        self.statuses = list(statuses)
        self.delay_s = delay_s
        self.hits: list[tuple[dict[str, str], bytes, str]] = []
        self._last = 500
        self._hits_lock = threading.Lock()
        self._server: http.server.ThreadingHTTPServer | None = None

    def __enter__(self) -> _Recorder:
        recorder = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — stdlib naming
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                with recorder._hits_lock:
                    recorder.hits.append(({k: v for k, v in self.headers.items()}, body, self.path))
                if recorder.delay_s:
                    # Event.wait — not time.sleep — so probes patching the
                    # client's backoff clock can't speed the server's delay
                    threading.Event().wait(recorder.delay_s)
                with recorder._hits_lock:
                    status = recorder.statuses.pop(0) if recorder.statuses else recorder._last
                    recorder._last = status
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *a: Any) -> None:  # silence
                return

        self._server = http.server.ThreadingHTTPServer((_LOOPBACK, 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    @property
    def url(self) -> str:
        assert self._server is not None
        port = self._server.server_address[1]
        return f"http://{_LOOPBACK}:{port}/cb?q=1"

    def __exit__(self, *a: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


# --------------------------------------------------------------------------
# probe sections
# --------------------------------------------------------------------------


def _probe_sign_verify() -> dict[str, bool]:
    """Byte-exact HMAC over ``<ts>.<raw body>`` plus the tolerance and
    shape guards around it."""
    out: dict[str, bool] = {}
    from fx1.serve.webhooks import sign_webhook, verify_webhook

    sig = sign_webhook(_SECRET, "1700000000", _BODY)
    out["sign_format_sha256_hex"] = (
        sig.startswith("sha256=")
        and len(sig) == len("sha256=") + 64
        and all(c in "0123456789abcdef" for c in sig[7:])
    )
    out["verify_roundtrip"] = verify_webhook(_SECRET, "1700000000", sig, _BODY, tolerance_s=-1)
    out["verify_rejects_wrong_secret"] = not verify_webhook(
        "other-secret", "1700000000", sig, _BODY, tolerance_s=-1
    )
    out["verify_rejects_tampered_body"] = not verify_webhook(
        _SECRET, "1700000000", sig, _BODY + b" ", tolerance_s=-1
    )
    out["verify_rejects_reserialized_body"] = not verify_webhook(
        _SECRET,
        "1700000000",
        sig,
        b'{"job_id": "j-1", "status": "succeeded", "result": {"ok": true}}',
        tolerance_s=-1,
    )
    out["verify_rejects_tampered_ts"] = not verify_webhook(
        _SECRET, "1700000001", sig, _BODY, tolerance_s=-1
    )
    out["verify_rejects_missing_ts"] = not verify_webhook(_SECRET, None, sig, _BODY, tolerance_s=-1)
    out["verify_rejects_missing_sig"] = not verify_webhook(
        _SECRET, "1700000000", None, _BODY, tolerance_s=-1
    )
    out["verify_rejects_empty_sig"] = not verify_webhook(
        _SECRET, "1700000000", "", _BODY, tolerance_s=-1
    )
    out["verify_rejects_bare_hex"] = not verify_webhook(
        _SECRET, "1700000000", sig[7:], _BODY, tolerance_s=-1
    )
    out["verify_rejects_wrong_prefix"] = not verify_webhook(
        _SECRET, "1700000000", "sha1=" + sig[7:], _BODY, tolerance_s=-1
    )
    out["verify_rejects_empty_secret"] = not verify_webhook(
        "", "1700000000", sig, _BODY, tolerance_s=-1
    )
    out["verify_rejects_nonascii_ts"] = not verify_webhook(
        _SECRET, "1700００００００", sig, _BODY, tolerance_s=-1
    )
    out["verify_rejects_nonascii_sig"] = not verify_webhook(
        _SECRET, "1700000000", "sha256=" + "ａ" * 64, _BODY, tolerance_s=-1
    )
    out["verify_never_raises_garbage"] = (
        verify_webhook(_SECRET, "not-a-float", "sha256=x", _BODY, tolerance_s=-1) is False
    )

    # freshness: inclusive boundary both directions, beyond is refused
    out["fresh_at_tolerance_edge_past"] = verify_webhook(
        _SECRET,
        "1000",
        sign_webhook(_SECRET, "1000", _BODY),
        _BODY,
        tolerance_s=300.0,
        now=1300.0,
    )
    out["fresh_at_tolerance_edge_future"] = verify_webhook(
        _SECRET,
        "1600",
        sign_webhook(_SECRET, "1600", _BODY),
        _BODY,
        tolerance_s=300.0,
        now=1300.0,
    )
    out["stale_beyond_tolerance_refused"] = not verify_webhook(
        _SECRET,
        "999",
        sign_webhook(_SECRET, "999", _BODY),
        _BODY,
        tolerance_s=300.0,
        now=1300.0,
    )
    out["future_beyond_tolerance_refused"] = not verify_webhook(
        _SECRET,
        "1601",
        sign_webhook(_SECRET, "1601", _BODY),
        _BODY,
        tolerance_s=300.0,
        now=1300.0,
    )
    out["signed_stale_still_refused"] = not verify_webhook(
        _SECRET, "100", sign_webhook(_SECRET, "100", _BODY), _BODY, now=10**9
    )
    out["negative_tolerance_disables_freshness"] = verify_webhook(
        _SECRET,
        "100",
        sign_webhook(_SECRET, "100", _BODY),
        _BODY,
        tolerance_s=-1.0,
        now=10**9,
    )
    out["nonfinite_tolerance_refused"] = not verify_webhook(
        _SECRET,
        "1000",
        sign_webhook(_SECRET, "1000", _BODY),
        _BODY,
        tolerance_s=float("inf"),
        now=1000.0,
    )
    out["nonfinite_ts_refused"] = not verify_webhook(
        _SECRET,
        "inf",
        sign_webhook(_SECRET, "inf", _BODY),
        _BODY,
        tolerance_s=-1,
    )
    out["nonfinite_now_refused"] = not verify_webhook(
        _SECRET,
        "1000",
        sign_webhook(_SECRET, "1000", _BODY),
        _BODY,
        tolerance_s=300.0,
        now=float("nan"),
    )
    import time as _time  # noqa: PLC0415

    now_s = str(int(_time.time()))
    out["verify_uses_wallclock_default"] = verify_webhook(
        _SECRET,
        now_s,
        sign_webhook(_SECRET, now_s, _BODY),
        _BODY,
    )
    # unicode + empty bodies sign fine — the bytes are the contract
    ubody = "données ✓".encode()
    usig = sign_webhook(_SECRET, "1700000000", ubody)
    out["unicode_body_roundtrip"] = verify_webhook(
        _SECRET, "1700000000", usig, ubody, tolerance_s=-1
    )
    esig = sign_webhook(_SECRET, "1700000000", b"")
    out["empty_body_roundtrip"] = verify_webhook(_SECRET, "1700000000", esig, b"", tolerance_s=-1)
    return out


def _probe_callback_url() -> dict[str, bool]:
    """``check_callback_url`` grammar: http(s)+host only, no userinfo,
    no port-0, and literal IPs get the SSRF classification up front."""
    out: dict[str, bool] = {}
    from fx1.serve.webhooks import check_callback_url

    def bad(url: Any) -> bool:
        try:
            check_callback_url(url)
        except ValueError:
            return True
        return False

    out["none_passes_through"] = check_callback_url(None) is None
    out["http_ok"] = check_callback_url("https://cb.example/hook") == ("https://cb.example/hook")
    out["scheme_ftp_refused"] = bad("ftp://cb.example/hook")
    out["scheme_ws_refused"] = bad("ws://cb.example/hook")
    out["scheme_file_refused"] = bad("file:///etc/passwd")
    out["no_host_refused"] = bad("https:///path-only")
    out["garbage_refused"] = bad("not a url")
    out["userinfo_refused"] = bad("https://user@cb.example/hook")
    out["userinfo_pass_refused"] = bad("https://user:pw@cb.example/hook")
    out["port_zero_refused"] = bad("https://cb.example:0/hook")
    out["port_garbage_refused"] = bad("https://cb.example:bad/hook")
    out["hostname_not_validated_without_dns"] = not bad("https://unresolvable-name.invalid/hook")

    literal_bad = [
        "http://127.0.0.1/h",  # NOSONAR(S1313)
        "http://10.0.0.1/h",  # NOSONAR(S1313)
        "http://192.168.1.1/h",  # NOSONAR(S1313)
        "http://172.16.0.1/h",  # NOSONAR(S1313) — RFC-1918 low edge
        "http://172.31.255.255/h",  # NOSONAR(S1313) — RFC-1918 high edge
        "http://100.64.0.1/h",  # NOSONAR(S1313) — CGNAT 100.64/10
        "http://100.127.255.254/h",  # NOSONAR(S1313) — CGNAT high edge
        "http://[fc00::1]/h",  # NOSONAR(S1313) — IPv6 ULA fc00::/7
        "http://[fd00::1]/h",  # NOSONAR(S1313) — IPv6 ULA fd00::/8
        "http://169.254.169.254/h",  # NOSONAR(S1313) — link-local metadata
        "http://0.0.0.0/h",  # NOSONAR(S1313)
        "http://224.0.0.1/h",  # NOSONAR(S1313) — multicast
        "http://[::1]/h",  # NOSONAR(S1313)
        "http://[fe80::1]/h",  # NOSONAR(S1313)
        "http://[::ffff:127.0.0.1]/h",  # NOSONAR(S1313) — v4-mapped unwrap
        "http://[::ffff:7f00:1]/h",  # NOSONAR(S1313) — hex-mapped loopback
    ]
    for i, url in enumerate(literal_bad):
        out[f"literal_private_refused_{i}"] = bad(url)

    with _allow_private():
        out["env_allows_loopback_literal"] = not bad("http://127.0.0.1/h")  # NOSONAR(S1313)
    out["env_restored_after_leg"] = bad("http://127.0.0.1/h")  # NOSONAR(S1313)

    # mid-process env changes are honored — no caching at import time
    prev = os.environ.get(_ENV)
    try:
        os.environ[_ENV] = "yes"
        out["env_yes_counts"] = not bad("http://127.0.0.1/h")  # NOSONAR(S1313)
        os.environ[_ENV] = "0"
        out["env_zero_does_not_count"] = bad("http://127.0.0.1/h")  # NOSONAR(S1313)
    finally:
        if prev is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = prev
    return out


def _probe_resolved_addresses() -> dict[str, bool]:
    """``_resolved_addresses`` — validate every getaddrinfo result,
    unwrap v4-mapped, dedupe, refuse the private list, error empty."""
    out: dict[str, bool] = {}
    from fx1.serve.webhooks import _resolved_addresses

    def fake_gai(results: list[tuple[Any, ...]]) -> Any:
        def gai(host: str, port: int, **kw: Any) -> list[tuple[Any, ...]]:
            return results

        return gai

    V4 = socket.AF_INET
    V6 = socket.AF_INET6
    TCP = socket.SOCK_STREAM
    UDP = socket.SOCK_DGRAM
    pub = [(V4, TCP, 6, "", ("93.184.216.34", 443))]

    with mock.patch.object(socket, "getaddrinfo", fake_gai(pub)):
        out["public_passthrough"] = _resolved_addresses("h.example", 443) == ("93.184.216.34",)

    dup = pub + [(V4, TCP, 6, "", ("93.184.216.34", 443))]
    with mock.patch.object(socket, "getaddrinfo", fake_gai(dup)):
        out["duplicates_deduped"] = _resolved_addresses("h.example", 443) == ("93.184.216.34",)

    # a genuinely mixed answer fails closed on the private member —
    # the public sibling does not rescue it
    mixed = pub + [(V4, TCP, 6, "", ("10.1.2.3", 443))]
    with mock.patch.object(socket, "getaddrinfo", fake_gai(mixed)):
        try:
            _resolved_addresses("h.example", 443)
            out["mixed_public_private_refused"] = False
        except ValueError:
            out["mixed_public_private_refused"] = True

    # the same special-use classes refused at resolution, not just as
    # literals — DNS answers get no lighter treatment
    for name, addr in (
        ("17216", "172.16.0.1"),
        ("cgnat", "100.64.0.1"),
        ("ula", "fd00::1"),
    ):
        fam = V6 if ":" in addr else V4
        sa = (addr, 443, 0, 0) if fam == V6 else (addr, 443)
        with mock.patch.object(socket, "getaddrinfo", fake_gai([(fam, TCP, 6, "", sa)])):
            try:
                _resolved_addresses("h.example", 443)
                out[f"resolved_{name}_refused"] = False
            except ValueError:
                out[f"resolved_{name}_refused"] = True

    # boundary sanity: 172.32.x is outside RFC-1918 — public, admitted
    edge = [(V4, TCP, 6, "", ("172.32.0.1", 443))]
    with mock.patch.object(socket, "getaddrinfo", fake_gai(edge)):
        out["resolved_17232_public_passes"] = _resolved_addresses("h.example", 443) == (
            "172.32.0.1",
        )

    priv = [(V4, TCP, 6, "", ("10.1.2.3", 443))]
    with mock.patch.object(socket, "getaddrinfo", fake_gai(priv)):
        try:
            _resolved_addresses("h.example", 443)
            out["private_resolution_refused"] = False
        except ValueError:
            out["private_resolution_refused"] = True

    with _allow_private(), mock.patch.object(socket, "getaddrinfo", fake_gai(priv)):
        out["private_allowed_under_env"] = _resolved_addresses("h.example", 443) == ("10.1.2.3",)

    mapped = [(V6, TCP, 6, "", ("::ffff:10.1.2.3", 443, 0, 0))]
    with mock.patch.object(socket, "getaddrinfo", fake_gai(mapped)):
        try:
            _resolved_addresses("h.example", 443)
            out["v4_mapped_private_refused"] = False
        except ValueError:
            out["v4_mapped_private_refused"] = True

    pubv6 = [(V6, TCP, 6, "", ("2606:2800:220:1:248:1893:25c8:1946", 443, 0, 0))]
    with mock.patch.object(socket, "getaddrinfo", fake_gai(pubv6)):
        out["public_v6_passthrough"] = _resolved_addresses("h.example", 443) == (
            "2606:2800:220:1:248:1893:25c8:1946",
        )

    non_tcp = [(V4, UDP, 17, "", ("93.184.216.34", 443))]
    with mock.patch.object(socket, "getaddrinfo", fake_gai(non_tcp)):
        try:
            _resolved_addresses("h.example", 443)
            out["non_tcp_skipped_then_oserror"] = False
        except OSError:
            out["non_tcp_skipped_then_oserror"] = True

    with mock.patch.object(socket, "getaddrinfo", fake_gai([])):
        try:
            _resolved_addresses("h.example", 443)
            out["empty_resolution_oserror"] = False
        except OSError:
            out["empty_resolution_oserror"] = True
    return out


def _probe_pinned_connection() -> dict[str, bool]:
    """The pin: TCP connects to the validated numeric address while the
    Host/SNI identity stays the URL host."""
    out: dict[str, bool] = {}
    from fx1.serve.webhooks import _PinnedHTTPConnection, _PinnedHTTPSConnection

    dialed: list[tuple[str, int]] = []
    fake_sock = mock.MagicMock(name="sock")

    def dial1(addr: tuple[str, int], *_a: Any) -> Any:
        dialed.append(addr)
        return fake_sock

    with mock.patch.object(socket, "create_connection", side_effect=dial1):
        conn = _PinnedHTTPConnection("h.example", 80, "93.184.216.34", 5.0)
        conn.connect()
    out["http_connect_dials_validated_ip"] = dialed == [("93.184.216.34", 80)]
    out["http_connection_keeps_url_host"] = conn.host == "h.example"

    dialed2: list[tuple[str, int]] = []
    snis: list[str | None] = []

    def dial2(addr: tuple[str, int], *_a: Any) -> Any:
        dialed2.append(addr)
        return fake_sock

    def wrap(sock: Any, server_hostname: str | None = None, **_k: Any) -> Any:
        snis.append(server_hostname)
        return sock

    conn2 = _PinnedHTTPSConnection("secure.example", 443, "93.184.216.35", 5.0)
    with (
        mock.patch.object(socket, "create_connection", side_effect=dial2),
        mock.patch.object(
            conn2._context,  # noqa: SLF001 — audit probes the pin seam directly
            "wrap_socket",
            side_effect=wrap,
        ),
    ):
        conn2.connect()
    out["https_connect_dials_validated_ip"] = dialed2 == [("93.184.216.35", 443)]
    out["https_sni_is_url_host"] = snis == ["secure.example"]
    return out


def _probe_deliver() -> dict[str, bool]:
    """``deliver_signed`` over a real loopback server: retry
    classification, backoff schedule, per-attempt freshness, address
    walk, and the never-raise contract."""
    out: dict[str, bool] = {}
    from fx1.serve.webhooks import deliver_signed, sign_webhook, verify_webhook

    with _allow_private(), _Recorder([500, 500, 200]) as rec:
        sleeps: list[float] = []
        with mock.patch("time.sleep", side_effect=lambda s: sleeps.append(s)):
            ok, err, attempts = deliver_signed(
                rec.url, _SECRET, _BODY, max_attempts=3, backoff_s=0.5
            )
        out["fail_fail_ok_delivers_third"] = ok and attempts == 3 and len(rec.hits) == 3
        out["backoff_doubles"] = sleeps == [0.5, 1.0]
        sigs = [h.get("X-Fx1-Webhook-Signature") for h, _, _ in rec.hits]
        tss = [h.get("X-Fx1-Webhook-Timestamp") for h, _, _ in rec.hits]
        out["signed_every_attempt"] = all(
            isinstance(s, str) and s.startswith("sha256=") for s in sigs
        ) and all(isinstance(t, str) and t.isdigit() for t in tss)
        out["each_attempt_verifies_on_raw_body"] = all(
            verify_webhook(_SECRET, t, s, b, tolerance_s=-1)
            for (_h, b, _p), s, t in zip(rec.hits, sigs, tss, strict=True)
        )

    with _allow_private(), _Recorder([404, 200]) as rec:
        ok, err, attempts = deliver_signed(rec.url, _SECRET, _BODY)
        out["definitive_4xx_no_retry"] = (
            not ok and attempts == 1 and len(rec.hits) == 1 and "404" in (err or "")
        )

    with _allow_private(), _Recorder([503]) as rec:
        sleeps2: list[float] = []
        with mock.patch("time.sleep", side_effect=lambda s: sleeps2.append(s)):
            ok, err, attempts = deliver_signed(rec.url, None, _BODY, max_attempts=3, backoff_s=0.25)
        out["persistent_5xx_exhausts"] = (
            not ok and attempts == 3 and len(rec.hits) == 3 and "503" in (err or "")
        )
        out["backoff_honors_custom_value"] = sleeps2 == [0.25, 0.5]
        out["unsigned_sends_no_signature"] = all(
            "X-Fx1-Webhook-Signature" not in h and "X-Fx1-Webhook-Timestamp" not in h
            for h, _, _ in rec.hits
        )
        out["body_reaches_every_attempt"] = all(b == _BODY for _, b, _ in rec.hits)

    # 2xx class: 204 counts as delivered
    with _allow_private(), _Recorder([204]) as rec:
        ok, _err, attempts = deliver_signed(rec.url, _SECRET, _BODY)
        out["status_204_delivered"] = ok and attempts == 1

    # 3xx is retryable (not <300 → error → not 4xx → outer retry)
    with _allow_private(), _Recorder([302, 200]) as rec:
        with mock.patch("time.sleep", lambda s: None):
            ok, _err, attempts = deliver_signed(rec.url, _SECRET, _BODY)
        out["redirect_is_retried_not_followed"] = ok and attempts == 2 and len(rec.hits) == 2

    # per-attempt freshness: cross a second boundary → new ts + new
    # sig. The fake clock keys off recorded hits (the server thread
    # burns time.time() for its Date header, so call counts lie).
    with _allow_private(), _Recorder([500, 200]) as rec:
        with (
            mock.patch("time.sleep", lambda s: None),
            mock.patch(
                "time.time",
                side_effect=lambda: 1700000001.1 if rec.hits else 1700000000.9,
            ),
        ):
            ok, _e, _a = deliver_signed(rec.url, _SECRET, _BODY)
        ts_pair = [h.get("X-Fx1-Webhook-Timestamp") for h, _, _ in rec.hits]
        sig_pair = [h.get("X-Fx1-Webhook-Signature") for h, _, _ in rec.hits]
        out["fresh_ts_per_attempt"] = ts_pair == ["1700000000", "1700000001"]
        out["new_second_new_signature"] = sig_pair[0] != sig_pair[1] and sig_pair[
            1
        ] == sign_webhook(_SECRET, "1700000001", _BODY)

    # same-second retry reuses the ts — identical (ts, body) ⇒ identical
    # signature: correct HMAC, not staleness. Both attempts floor to
    # the same second even with the Date-header burn in between.
    with _allow_private(), _Recorder([500, 200]) as rec:
        with (
            mock.patch("time.sleep", lambda s: None),
            mock.patch(
                "time.time",
                side_effect=lambda: 1700000002.9 if rec.hits else 1700000002.1,
            ),
        ):
            deliver_signed(rec.url, _SECRET, _BODY)
        sig_pair2 = [h.get("X-Fx1-Webhook-Signature") for h, _, _ in rec.hits]
        out["same_second_same_signature"] = sig_pair2[0] == sig_pair2[1]

    # connection refused: all attempts fault → error string, never raise
    dead = socket.socket()
    dead.bind((_LOOPBACK, 0))
    dead_port = dead.getsockname()[1]
    dead.close()
    with _allow_private():
        with mock.patch("time.sleep", lambda s: None):
            ok, err, attempts = deliver_signed(
                f"http://{_LOOPBACK}:{dead_port}/x", _SECRET, _BODY, max_attempts=3
            )
        out["conn_refused_retries_then_errors"] = (
            not ok and attempts == 3 and isinstance(err, str) and len(err) > 0
        )

    # invalid targets return (False, msg, 0) — never raise
    bad_urls: list[Any] = [None, "", "not a url", "ftp://x/h", "https://u:p@h/cb"]
    for i, url in enumerate(bad_urls):
        ok, err, attempts = deliver_signed(url, _SECRET, _BODY)
        out[f"invalid_url_never_raises_{i}"] = not ok and attempts == 0 and isinstance(err, str)

    # private target without env → refused before any dial
    ok, err, attempts = deliver_signed("http://127.0.0.1:9/h", _SECRET, _BODY)  # NOSONAR(S1313)
    out["private_url_refused_zero_attempts"] = (
        not ok and attempts == 0 and "ValueError" in (err or "")
    )

    # a slow endpoint runs past timeout_s → fault retried, never raised
    with _allow_private(), _Recorder([200], delay_s=0.4) as rec:
        with mock.patch("time.sleep", lambda s: None):
            ok, err, attempts = deliver_signed(
                rec.url, _SECRET, _BODY, max_attempts=2, timeout_s=0.15
            )
        out["slow_endpoint_times_out_retries"] = not ok and attempts == 2 and isinstance(err, str)
    return out


def _probe_deliver_address_walk() -> dict[str, bool]:
    """Multi-address walks: exceptions try every validated address,
    a 5xx from one address short-circuits the rest of the list, and a
    definitive 4xx ends the walk early."""
    out: dict[str, bool] = {}
    from fx1.serve.webhooks import deliver_signed

    V4 = socket.AF_INET
    TCP = socket.SOCK_STREAM

    def gai_for(addrs: list[str]) -> Any:
        def gai(host: str, port: int, **kw: Any) -> list[tuple[Any, ...]]:
            return [(V4, TCP, 6, "", (a, port)) for a in addrs]

        return gai

    posted_to: list[str] = []
    faults_on: set[str] = set()
    scripted_status: dict[str, int] = {}

    def fake_post(
        parsed: Any, address: str, body: bytes, headers: dict[str, str], timeout_s: float
    ) -> int:
        posted_to.append(address)
        if address in faults_on:
            raise OSError("dial failed")
        return scripted_status.get(address, 500)

    url = "http://93.184.216.34/cb"  # NOSONAR(S1313) — never dialed: _post_once stubbed
    with (
        _allow_private(),
        mock.patch("fx1.serve.webhooks._post_once", side_effect=fake_post),
        mock.patch("time.sleep", lambda s: None),
        mock.patch.object(socket, "getaddrinfo", gai_for(["93.184.216.34", "93.184.216.35"])),
    ):
        # first address faults → second is tried
        faults_on.add("93.184.216.34")
        scripted_status["93.184.216.35"] = 200
        ok, _e, att = deliver_signed(url, "s", b"b")
        out["fault_walks_to_next_address"] = (
            ok
            and posted_to
            == [
                "93.184.216.34",
                "93.184.216.35",
            ]
            and att == 1
        )

        posted_to.clear()
        faults_on.clear()
        scripted_status["93.184.216.34"] = 503
        scripted_status["93.184.216.35"] = 200
        ok, _e, att = deliver_signed(url, "s", b"b", max_attempts=1)
        out["five_xx_short_circuits_walk"] = not ok and posted_to == ["93.184.216.34"] and att == 1

        posted_to.clear()
        scripted_status["93.184.216.34"] = 404
        ok, _e, att = deliver_signed(url, "s", b"b", max_attempts=1)
        out["four_xx_ends_walk_definitively"] = (
            not ok and posted_to == ["93.184.216.34"] and att == 1
        )

        posted_to.clear()
        faults_on.update({"93.184.216.34", "93.184.216.35"})
        ok, err, att = deliver_signed(url, "s", b"b", max_attempts=2)
        out["all_addresses_fault_retries_outer"] = (
            not ok and posted_to == ["93.184.216.34", "93.184.216.35"] * 2 and att == 2
        )
        out["walk_fault_error_names_exception"] = "OSError" in (err or "")
    return out


def _probe_deliver_misc() -> dict[str, bool]:
    """Remaining measured edges: path/query preservation, content-type,
    max_attempts honored, and the resolution-failure path."""
    out: dict[str, bool] = {}
    from fx1.serve.webhooks import deliver_signed

    with _allow_private(), _Recorder([200]) as rec:
        deliver_signed(rec.url, "s", _BODY)
        h, b, path = rec.hits[0]
        out["content_type_json"] = h.get("Content-Type") == "application/json"
        out["content_length_matches_body"] = h.get("Content-Length") == str(len(_BODY))
        out["path_and_query_reach_wire"] = path == "/cb?q=1"

    # target construction is path-or-slash + query verbatim
    import urllib.parse

    parsed = urllib.parse.urlparse("http://h.example:8080/a/b?x=1&y=2")
    target = urllib.parse.urlunparse(("", "", parsed.path or "/", parsed.params, parsed.query, ""))
    out["target_preserves_path_query"] = target == "/a/b?x=1&y=2"
    parsed_root = urllib.parse.urlparse("http://h.example")
    target_root = urllib.parse.urlunparse(
        ("", "", parsed_root.path or "/", parsed_root.params, parsed_root.query, "")
    )
    out["root_target_is_slash"] = target_root == "/"

    # max_attempts=1 honored; max_attempts=0 never dials
    with _allow_private(), _Recorder([500]) as rec:
        with mock.patch("time.sleep", lambda s: None):
            ok, err, att = deliver_signed(rec.url, "s", _BODY, max_attempts=1)
        out["max_attempts_one_single_call"] = not ok and att == 1 and len(rec.hits) == 1
    with _allow_private():
        ok, err, att = deliver_signed(
            "http://127.0.0.1:9/x", "s", _BODY, max_attempts=0
        )  # NOSONAR(S1313)
        out["max_attempts_zero_never_dials"] = not ok and att == 0

    # resolution failure counts as a per-attempt fault
    with (
        _allow_private(),
        mock.patch("time.sleep", lambda s: None),
        mock.patch.object(socket, "getaddrinfo", side_effect=socket.gaierror("no such host")),
    ):
        ok, err, att = deliver_signed(
            "http://gone.invalid/cb", "s", _BODY, max_attempts=2
        )  # NOSONAR(S1313)
        out["resolution_fault_retries_and_errors"] = (
            not ok and att == 2 and "gaierror" in (err or "")
        )
    return out


# --------------------------------------------------------------------------
# battery + bench
# --------------------------------------------------------------------------


def webhook_delivery_audit() -> dict[str, bool]:
    """The full webhook-delivery battery. Scripted server + patched
    clocks — deterministic, no external network."""
    with _audit_context():
        results: dict[str, bool] = {}
        results.update(_probe_sign_verify())
        results.update(_probe_callback_url())
        results.update(_probe_resolved_addresses())
        results.update(_probe_pinned_connection())
        results.update(_probe_deliver())
        results.update(_probe_deliver_address_walk())
        results.update(_probe_deliver_misc())
    missing = _EXPECTED_PROBES - results.keys()
    extra = results.keys() - _EXPECTED_PROBES
    if missing or extra:
        raise AssertionError(
            f"battery drifted from the pinned probe set — missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    return results


def webhook_delivery_audit_bench(results: dict[str, bool] | None = None) -> dict[str, Any]:
    """Seal webhook-delivery results; run the battery when omitted."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = webhook_delivery_audit() if results is None else dict(results)
    ok = set(r) == _EXPECTED_PROBES and all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "webhook_delivery_audit",
        "schema": "webhook_delivery_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "scripted loopback http.server + stubbed sockets/DNS",
            "not_verified": [
                "real TLS peer validation (SNI is pinned, handshake is stubbed)",
                "cross-second real-clock signature drift",
                "response-body cap behavior (deliveries never read the body)",
                "delivery from the api.py worker path (route-level pins live in webhook_audit)",
            ],
        },
        "interpretation": (
            "Webhook delivery holds: HMAC-sha256 over '<ts>.<raw body>' "
            "verifies byte-exactly and rejects tamper/reserialization/"
            "replay past tolerance (inclusive boundary, negative "
            "tolerance disables freshness); callback URLs validate to "
            "http(s)+host with no userinfo and refuse special-use IP "
            "literals unless the private-networks env is set mid-process; "
            "resolution validates every getaddrinfo result, unwraps "
            "v4-mapped, dedupes, and errors empty; the pinned connection "
            "dials the validated IP while TLS authenticates the URL "
            "host; deliver_signed retries transport faults and non-4xx "
            "statuses with doubling backoff, walks remaining addresses "
            "only on faults (a 5xx short-circuits the list), treats 4xx "
            "as definitive, re-signs per attempt with a fresh timestamp, "
            "and never raises — returning (ok, error, attempts). "
            "SYNTHETIC scripted deliveries only — no research claim."
            if ok
            else f"WEBHOOK DELIVERY AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(webhook_delivery_audit_bench(), indent=1))
