"""SYNTHETIC webhook destination-policy probes; no external requests or API imports."""

from __future__ import annotations

import ipaddress
import os
import socket
import unittest
from unittest.mock import patch

from fx1.serve import webhooks

_LEGACY_ADDRESSES = (
    "192.88.99.0",
    "192.88.99.1",
    "192.88.99.2",
    "192.88.99.255",
    "::ffff:192.88.99.0",
    "::ffff:192.88.99.1",
    "::ffff:192.88.99.2",
    "::ffff:192.88.99.255",
    "::ffff:c058:6302",
    "fec0::",
    "fec0::1",
    "feff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
)


def _url(raw: str) -> str:
    host = f"[{raw}]" if ":" in raw else raw
    return f"https://{host}/synthetic-hook"


def _answer(raw: str) -> tuple[int, int, int, str, tuple[str, int]]:
    family = socket.AF_INET if ipaddress.ip_address(raw).version == 4 else socket.AF_INET6
    return family, socket.SOCK_STREAM, 6, "", (raw, 443)


class TestWebhookSpecialAddresses(unittest.TestCase):
    def test_legacy_literals_are_refused_before_dns_in_both_modes(self) -> None:
        for raw in _LEGACY_ADDRESSES:
            for opt_in in ("", "1"):
                with (
                    self.subTest(raw=raw, opt_in=opt_in),
                    patch.dict(os.environ, {"FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS": opt_in}),
                    patch.object(socket, "getaddrinfo") as resolve,
                    self.assertRaisesRegex(ValueError, "deprecated special-use"),
                ):
                    webhooks.check_callback_url(_url(raw))
                resolve.assert_not_called()

    def test_legacy_literal_delivery_never_resolves_or_connects(self) -> None:
        for raw in _LEGACY_ADDRESSES:
            for opt_in in ("", "1"):
                with (
                    self.subTest(raw=raw, opt_in=opt_in),
                    patch.dict(os.environ, {"FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS": opt_in}),
                    patch.object(socket, "getaddrinfo") as resolve,
                    patch.object(socket, "create_connection") as connect,
                ):
                    delivered, error, attempts = webhooks.deliver_signed(
                        _url(raw), None, b"{}", max_attempts=1, backoff_s=0
                    )
                    self.assertFalse(delivered)
                    self.assertIn("deprecated special-use", error or "")
                    self.assertEqual(attempts, 0)
                    resolve.assert_not_called()
                    connect.assert_not_called()

    def test_legacy_and_mixed_dns_answers_are_refused_before_connect(self) -> None:
        public = _answer("93.184.216.34")
        for raw in _LEGACY_ADDRESSES:
            denied = _answer(raw)
            for answers in ([denied], [public, denied], [denied, public]):
                for opt_in in ("", "1"):
                    with (
                        self.subTest(raw=raw, answers=answers, opt_in=opt_in),
                        patch.dict(os.environ, {"FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS": opt_in}),
                        patch.object(socket, "getaddrinfo", return_value=answers),
                        patch.object(socket, "create_connection") as connect,
                    ):
                        delivered, error, attempts = webhooks.deliver_signed(
                            "https://callback.example/synthetic-hook",
                            None,
                            b"{}",
                            max_attempts=1,
                            backoff_s=0,
                        )
                        self.assertFalse(delivered)
                        self.assertIn("deprecated special-use", error or "")
                        self.assertEqual(attempts, 1)
                        connect.assert_not_called()

    def test_public_addresses_keep_their_numeric_family_and_scope(self) -> None:
        for raw in (
            "93.184.216.34",
            "192.88.98.255",
            "192.88.100.0",
            "2606:4700:4700::1111",
            "::ffff:93.184.216.34",
            "::ffff:192.88.98.255",
            "::ffff:192.88.100.0",
        ):
            for opt_in in ("", "1"):
                with (
                    self.subTest(raw=raw, opt_in=opt_in),
                    patch.dict(os.environ, {"FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS": opt_in}),
                    patch.object(socket, "getaddrinfo", return_value=[_answer(raw)]),
                ):
                    self.assertEqual(webhooks.check_callback_url(_url(raw)), _url(raw))
                    self.assertEqual(
                        webhooks._resolved_addresses("callback.example", 443),
                        (str(ipaddress.ip_address(raw)),),
                    )

    def test_private_opt_in_behavior_is_preserved(self) -> None:
        for raw in ("10.1.2.3", "127.0.0.1", "::1", "fd00::1", "::ffff:10.1.2.3"):
            for opt_in in ("", "1"):
                with (
                    self.subTest(raw=raw, opt_in=opt_in),
                    patch.dict(os.environ, {"FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS": opt_in}),
                    patch.object(socket, "getaddrinfo", return_value=[_answer(raw)]),
                ):
                    if opt_in:
                        self.assertEqual(webhooks.check_callback_url(_url(raw)), _url(raw))
                        self.assertEqual(
                            webhooks._resolved_addresses("callback.example", 443),
                            (str(ipaddress.ip_address(raw)),),
                        )
                    else:
                        with self.assertRaisesRegex(ValueError, "private or special-use"):
                            webhooks.check_callback_url(_url(raw))
                        with self.assertRaisesRegex(ValueError, "private or special-use"):
                            webhooks._resolved_addresses("callback.example", 443)
