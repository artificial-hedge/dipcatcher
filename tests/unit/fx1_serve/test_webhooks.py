"""SYNTHETIC webhook authentication checks; no API, callbacks, or receipts."""

from __future__ import annotations

import hashlib
import hmac
import http.client
import io
import os
import socket
import unittest
import urllib.parse
from typing import Any
from unittest.mock import Mock, patch

from fx1.serve.webhooks import (
    _post_once,
    check_callback_url,
    deliver_signed,
    sign_webhook,
    verify_webhook,
)


class TestWebhooks(unittest.TestCase):
    secret = "synthetic-secret"
    body = b'{"status":"synthetic"}'

    def signature(self, timestamp: str) -> str:
        # Independently construct the wire signature to catch signing drift.
        payload = timestamp.encode("ascii") + b"." + self.body
        digest = hmac.new(self.secret.encode(), payload, hashlib.sha256).hexdigest()
        return f"sha256={digest}"

    def test_valid_signature_and_exact_body(self) -> None:
        signature = self.signature("1000")
        self.assertEqual(sign_webhook(self.secret, "1000", self.body), signature)
        self.assertTrue(verify_webhook(self.secret, "1000", signature, self.body, now=1000))
        self.assertFalse(verify_webhook("wrong", "1000", signature, self.body, now=1000))
        self.assertFalse(verify_webhook(self.secret, "1000", signature, b"{}", now=1000))
        self.assertFalse(verify_webhook(self.secret, "1001", signature, self.body, now=1000))

    def test_freshness_boundaries(self) -> None:
        for timestamp, accepted in (
            ("700", True),
            ("1300", True),
            ("699.999", False),
            ("1300.001", False),
            ("1000.25", True),
        ):
            with self.subTest(timestamp=timestamp):
                self.assertEqual(
                    verify_webhook(
                        self.secret, timestamp, self.signature(timestamp), self.body, now=1000
                    ),
                    accepted,
                )

    def test_zero_and_negative_tolerance(self) -> None:
        signature = self.signature("1000")
        self.assertTrue(
            verify_webhook(self.secret, "1000", signature, self.body, tolerance_s=0, now=1000)
        )
        self.assertFalse(
            verify_webhook(self.secret, "1000", signature, self.body, tolerance_s=0, now=1000.001)
        )
        self.assertTrue(
            verify_webhook(self.secret, "1000", signature, self.body, tolerance_s=-1, now=1000000)
        )

    def test_disabled_freshness_ignores_clock(self) -> None:
        self.assertTrue(
            verify_webhook(
                self.secret,
                "1000",
                self.signature("1000"),
                self.body,
                tolerance_s=-1,
                now=float("nan"),
            )
        )

    def test_utf8_secret(self) -> None:
        secret = "synthetic-é"
        signature = sign_webhook(secret, "1000", self.body)
        self.assertTrue(verify_webhook(secret, "1000", signature, self.body, now=1000))

    def test_malformed_runtime_types(self) -> None:
        overrides: list[tuple[str, Any]] = [
            ("secret", None),
            ("secret", b"secret"),
            ("timestamp", 1000),
            ("timestamp", b"1000"),
            ("signature", b"sha256=00"),
            ("body", None),
            ("body", "{}"),
            ("now", "1000"),
            ("now", 10**10000),
            ("tolerance_s", None),
            ("tolerance_s", "300"),
            ("tolerance_s", 10**10000),
        ]
        for field, value in overrides:
            with self.subTest(field=field, value_type=type(value).__name__):
                inputs: dict[str, Any] = {
                    "secret": self.secret,
                    "timestamp": "1000",
                    "signature": self.signature("1000"),
                    "body": self.body,
                    "now": 1000,
                }
                inputs[field] = value
                self.assertFalse(verify_webhook(**inputs))

    def test_default_clock(self) -> None:
        with patch("time.time", return_value=1000):
            self.assertTrue(verify_webhook(self.secret, "1000", self.signature("1000"), self.body))
            self.assertFalse(verify_webhook(self.secret, "699", self.signature("699"), self.body))
        with patch("time.time", return_value=float("nan")):
            self.assertFalse(verify_webhook(self.secret, "1000", self.signature("1000"), self.body))

    def test_nonfinite_timestamps_rejected_even_without_freshness_check(self) -> None:
        for timestamp in ("nan", "NaN", "inf", "+Infinity", "-inf", "1e9999"):
            for tolerance in (300, -1):
                with self.subTest(timestamp=timestamp, tolerance=tolerance):
                    self.assertFalse(
                        verify_webhook(
                            self.secret,
                            timestamp,
                            self.signature(timestamp),
                            self.body,
                            now=1000,
                            tolerance_s=tolerance,
                        )
                    )

    def test_invalid_clock_and_tolerance(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assertFalse(
                    verify_webhook(
                        self.secret, "1000", self.signature("1000"), self.body, now=value
                    )
                )
                self.assertFalse(
                    verify_webhook(
                        self.secret,
                        "1000",
                        self.signature("1000"),
                        self.body,
                        now=1000,
                        tolerance_s=value,
                    )
                )

    def test_malformed_headers(self) -> None:
        for timestamp in (None, "", "garbage", "١٠٠٠", "１０００", "1000\ud800"):
            with self.subTest(timestamp=timestamp):
                self.assertFalse(
                    verify_webhook(
                        self.secret, timestamp, self.signature("1000"), self.body, now=1000
                    )
                )
        for signature in (None, "", "sha1=00", "sha256=", "sha256=é", "sha256=\ud800"):
            with self.subTest(signature=signature):
                self.assertFalse(
                    verify_webhook(self.secret, "1000", signature, self.body, now=1000)
                )

    def test_malformed_secret(self) -> None:
        for secret in ("", "synthetic-\ud800"):
            with self.subTest(secret=secret):
                self.assertFalse(
                    verify_webhook(secret, "1000", self.signature("1000"), self.body, now=1000)
                )

    def test_private_and_special_use_ip_literals_are_rejected(self) -> None:
        rejected = (
            "http://127.0.0.1/hook",
            "http://10.0.0.1/hook",
            "http://169.254.169.254/latest/meta-data/",
            "http://100.64.0.1/hook",
            "http://192.0.2.1/hook",
            "http://[::1]/hook",
            "http://[fc00::1]/hook",
            "http://224.0.0.1/hook",
            "http://0.0.0.0/hook",
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", None)
            for url in rejected:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    check_callback_url(url)
        self.assertEqual(check_callback_url("https://8.8.8.8/hook"), "https://8.8.8.8/hook")

    def test_private_network_opt_in_is_explicit(self) -> None:
        with patch.dict(os.environ, {"FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS": "1"}):
            self.assertEqual(
                check_callback_url("http://127.0.0.1/hook"),
                "http://127.0.0.1/hook",
            )

    def test_invalid_port_is_rejected_without_delivery_attempt(self) -> None:
        delivered, error, attempts = deliver_signed(
            "https://example.com:bad/hook",
            None,
            self.body,
        )
        self.assertFalse(delivered)
        self.assertIn("invalid port", error or "")
        self.assertEqual(attempts, 0)

    def test_dns_answers_are_validated_before_connecting(self) -> None:
        answers = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ]
        with (
            patch("fx1.serve.webhooks.socket.getaddrinfo", return_value=answers),
            patch("fx1.serve.webhooks._post_once") as post,
        ):
            delivered, error, attempts = deliver_signed(
                "https://example.com/hook",
                self.secret,
                self.body,
                max_attempts=1,
                backoff_s=0,
            )
        self.assertFalse(delivered)
        self.assertIn("private or special-use", error or "")
        self.assertEqual(attempts, 1)
        post.assert_not_called()

    def test_delivery_connects_to_the_validated_numeric_address(self) -> None:
        answers = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
        ]
        with (
            patch("fx1.serve.webhooks.socket.getaddrinfo", return_value=answers),
            patch("fx1.serve.webhooks._post_once", return_value=204) as post,
        ):
            delivered, error, attempts = deliver_signed(
                "https://example.com/hook?x=1",
                self.secret,
                self.body,
                max_attempts=1,
                backoff_s=0,
            )
        self.assertTrue(delivered)
        self.assertIsNone(error)
        self.assertEqual(attempts, 1)
        self.assertEqual(post.call_args.args[1], "93.184.216.34")

    def test_dns_is_revalidated_on_every_retry(self) -> None:
        public = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
        ]
        private = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ]
        with (
            patch(
                "fx1.serve.webhooks.socket.getaddrinfo",
                side_effect=[public, private],
            ) as resolve,
            patch("fx1.serve.webhooks._post_once", return_value=503) as post,
        ):
            delivered, error, attempts = deliver_signed(
                "https://example.com/hook",
                None,
                self.body,
                max_attempts=2,
                backoff_s=0,
            )
        self.assertFalse(delivered)
        self.assertIn("private or special-use", error or "")
        self.assertEqual(attempts, 2)
        self.assertEqual(resolve.call_count, 2)
        post.assert_called_once()

    def test_post_once_closes_real_responses_without_reading_the_body(self) -> None:
        for scheme, port, connection_name in (
            ("http", 80, "_PinnedHTTPConnection"),
            ("https", 443, "_PinnedHTTPSConnection"),
        ):
            for status in (200, 204, 302, 400, 503):
                for framing in (
                    "Content-Length: 1000000000",
                    "Transfer-Encoding: chunked",
                    "Connection: close",
                ):
                    with self.subTest(scheme=scheme, status=status, framing=framing):
                        stream = io.BytesIO(
                            f"HTTP/1.1 {status} Synthetic\r\n{framing}\r\n\r\n".encode()
                        )
                        sock = Mock()
                        sock.makefile.return_value = stream
                        response = http.client.HTTPResponse(sock)
                        response.begin()
                        headers = {"Content-Type": "application/json"}
                        with (
                            patch(f"fx1.serve.webhooks.{connection_name}") as connection_cls,
                            patch.object(
                                response,
                                "read",
                                side_effect=AssertionError("body must not be read"),
                            ) as read,
                            patch.object(response, "close", wraps=response.close) as close,
                        ):
                            connection = connection_cls.return_value
                            connection.getresponse.return_value = response
                            self.assertEqual(
                                _post_once(
                                    urllib.parse.urlparse(f"{scheme}://example.com/hook?x=1"),
                                    "93.184.216.34",
                                    self.body,
                                    headers,
                                    10.0,
                                ),
                                status,
                            )
                            connection_cls.assert_called_once_with(
                                "example.com", port, "93.184.216.34", 10.0
                            )
                            connection.request.assert_called_once_with(
                                "POST", "/hook?x=1", body=self.body, headers=headers
                            )
                            read.assert_not_called()
                            close.assert_called_once_with()
                            connection.close.assert_called_once_with()
                        self.assertTrue(response.isclosed())
                        self.assertTrue(stream.closed)

    def test_post_once_closes_connection_on_request_or_response_failure(self) -> None:
        for operation in ("request", "getresponse"):
            with (
                self.subTest(operation=operation),
                patch("fx1.serve.webhooks._PinnedHTTPConnection") as connection_cls,
            ):
                connection = connection_cls.return_value
                getattr(connection, operation).side_effect = OSError("synthetic failure")
                with self.assertRaisesRegex(OSError, "synthetic failure"):
                    _post_once(
                        urllib.parse.urlparse("http://example.com/hook"),
                        "93.184.216.34",
                        self.body,
                        {},
                        10.0,
                    )
                connection.close.assert_called_once_with()

    def test_post_once_closes_connection_when_response_close_fails(self) -> None:
        sock = Mock()
        sock.makefile.return_value = io.BytesIO(b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n")
        response = http.client.HTTPResponse(sock)
        response.begin()
        try:
            with (
                patch("fx1.serve.webhooks._PinnedHTTPConnection") as connection_cls,
                patch.object(response, "close", side_effect=OSError("synthetic close failure")),
                patch.object(response, "read", side_effect=AssertionError("body must not be read")),
            ):
                connection = connection_cls.return_value
                connection.getresponse.return_value = response
                with self.assertRaisesRegex(OSError, "synthetic close failure"):
                    _post_once(
                        urllib.parse.urlparse("http://example.com/hook"),
                        "93.184.216.34",
                        self.body,
                        {},
                        10.0,
                    )
                connection.close.assert_called_once_with()
        finally:
            response.close()


if __name__ == "__main__":
    unittest.main()
