"""SYNTHETIC BYOK TLS checks; no live connections, API, or research imports."""

from __future__ import annotations

import ssl
import unittest
from unittest.mock import Mock, patch

from fx1.serve.backends import _ByokPinnedHTTPSConnection


class TestByokTLS(unittest.TestCase):
    def test_context_advertises_only_http11_and_retains_verification(self) -> None:
        context = ssl.create_default_context()
        settings = (
            "verify_mode",
            "check_hostname",
            "minimum_version",
            "maximum_version",
            "options",
            "verify_flags",
            "post_handshake_auth",
        )
        original = {name: getattr(context, name) for name in settings}
        with (
            patch("fx1.serve.backends.ssl.create_default_context", return_value=context) as create,
            patch.object(context, "set_alpn_protocols", wraps=context.set_alpn_protocols) as alpn,
            patch(
                "fx1.serve.backends.socket.create_connection",
                side_effect=AssertionError("constructing a connection must not open a socket"),
            ) as connect,
        ):
            connection = _ByokPinnedHTTPSConnection("provider.example", 443, "93.184.216.34", 2)
        create.assert_called_once_with()
        alpn.assert_called_once_with(["http/1.1"])
        connect.assert_not_called()
        self.assertIs(connection._context, context)
        self.assertIsNone(connection.sock)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertEqual({name: getattr(context, name) for name in settings}, original)

    def test_connect_preserves_pinned_address_and_authenticated_hostname(self) -> None:
        for address, port in (("93.184.216.34", 443), ("2606:4700:4700::1111", 8443)):
            with self.subTest(address=address, port=port):
                context = ssl.create_default_context()
                raw_socket = Mock()
                tls_socket = Mock()
                with (
                    patch("fx1.serve.backends.ssl.create_default_context", return_value=context),
                    patch(
                        "fx1.serve.backends.socket.create_connection", return_value=raw_socket
                    ) as connect,
                    patch.object(context, "wrap_socket", return_value=tls_socket) as wrap,
                ):
                    connection = _ByokPinnedHTTPSConnection("provider.example", port, address, 2)
                    connection.connect()
                connect.assert_called_once_with((address, port), 2, None)
                wrap.assert_called_once_with(raw_socket, server_hostname="provider.example")
                self.assertIs(connection.sock, tls_socket)
                self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
                self.assertTrue(context.check_hostname)
                connection.close()
                tls_socket.close.assert_called_once_with()

    def test_tls_failure_closes_raw_socket_and_propagates(self) -> None:
        context = ssl.create_default_context()
        raw_socket = Mock()
        failure = ssl.SSLCertVerificationError("synthetic certificate rejection")
        with (
            patch("fx1.serve.backends.ssl.create_default_context", return_value=context),
            patch(
                "fx1.serve.backends.socket.create_connection", return_value=raw_socket
            ) as connect,
            patch.object(context, "wrap_socket", side_effect=failure) as wrap,
        ):
            connection = _ByokPinnedHTTPSConnection("provider.example", 443, "93.184.216.34", 2)
            with self.assertRaises(ssl.SSLCertVerificationError) as raised:
                connection.connect()
        self.assertIs(raised.exception, failure)
        self.assertIsNone(connection.sock)
        connect.assert_called_once_with(("93.184.216.34", 443), 2, None)
        wrap.assert_called_once_with(raw_socket, server_hostname="provider.example")
        raw_socket.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
