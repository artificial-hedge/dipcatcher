"""SYNTHETIC BYOK special-address checks; no live requests or research imports."""

from __future__ import annotations

import ipaddress
import socket
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from fx1.serve import backends


class TestByokSpecialAddresses(unittest.TestCase):
    def test_legacy_special_use_ranges_are_denied_in_both_modes(self) -> None:
        for raw in (
            "192.88.99.0",
            "192.88.99.1",
            "192.88.99.2",  # Active 6a44 assignment is not globally reachable either.
            "192.88.99.255",
            "::ffff:192.88.99.1",
            "::ffff:192.88.99.2",
            "fec0::",
            "fec0::1",
            "feff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
        ):
            for allow_private in (False, True):
                with self.subTest(raw=raw, allow_private=allow_private):
                    self.assertFalse(
                        backends._byok_address_allowed(
                            backends._normalized_address(raw), allow_private=allow_private
                        )
                    )

    def test_public_and_narrow_private_policies_are_preserved(self) -> None:
        for raw in (
            "93.184.216.34",
            "192.88.98.255",
            "192.88.100.0",
            "2606:4700:4700::1111",
            "::ffff:93.184.216.34",
        ):
            for allow_private in (False, True):
                with self.subTest(raw=raw, allow_private=allow_private):
                    self.assertTrue(
                        backends._byok_address_allowed(
                            backends._normalized_address(raw), allow_private=allow_private
                        )
                    )
        for raw in ("10.1.2.3", "127.0.0.1", "172.16.0.1", "192.168.1.1", "::1", "fd00::1"):
            for allow_private in (False, True):
                with self.subTest(raw=raw, allow_private=allow_private):
                    self.assertEqual(
                        backends._byok_address_allowed(
                            backends._normalized_address(raw), allow_private=allow_private
                        ),
                        allow_private,
                    )
        for raw in ("169.254.169.254", "100.64.0.1", "192.0.2.1", "fe80::1", "::", "ff02::1"):
            for allow_private in (False, True):
                with self.subTest(raw=raw, allow_private=allow_private):
                    self.assertFalse(
                        backends._byok_address_allowed(
                            backends._normalized_address(raw), allow_private=allow_private
                        )
                    )

    def test_special_and_mixed_dns_answers_refuse_before_connect(self) -> None:
        for raw in ("192.88.99.1", "192.88.99.2", "::ffff:192.88.99.1", "fec0::1"):
            family = socket.AF_INET if ipaddress.ip_address(raw).version == 4 else socket.AF_INET6
            prohibited = (family, socket.SOCK_STREAM, 6, "", (raw, 443))
            public = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
            for answers in ([prohibited], [public, prohibited], [prohibited, public]):
                for allow_private in (False, True):
                    with self.subTest(raw=raw, answers=answers, allow_private=allow_private):
                        with (
                            patch.object(socket, "getaddrinfo", return_value=answers),
                            patch.object(
                                socket,
                                "create_connection",
                                side_effect=AssertionError(
                                    "prohibited destination reached connect"
                                ),
                            ) as connect,
                            self.assertRaises(urllib.error.URLError) as raised,
                        ):
                            backends._open_byok_pinned(
                                urllib.request.Request(
                                    "https://provider.example/v1/chat/completions",
                                    data=b"{}",
                                    method="POST",
                                ),
                                timeout_s=1,
                                allow_private=allow_private,
                            )
                        self.assertIsInstance(raised.exception.reason, ValueError)
                        self.assertIn("prohibited network address", str(raised.exception.reason))
                        connect.assert_not_called()
