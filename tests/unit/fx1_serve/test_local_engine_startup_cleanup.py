"""SYNTHETIC startup lifecycle checks: no real child processes or engines."""

from __future__ import annotations

import io
import json
import subprocess
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fx1.serve import backends


class _BoundedLock:
    """A real non-reentrant lock that fails a regression instead of hanging."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def __enter__(self) -> _BoundedLock:
        if not self._lock.acquire(timeout=0.1):
            raise AssertionError("startup cleanup reacquired the held engine lock")
        return self

    def __exit__(self, *exc: object) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


class _SyntheticProcess:
    def __init__(self, lock: _BoundedLock, returncode: int | None = None) -> None:
        self.lock = lock
        self.returncode = returncode
        self.calls: list[tuple[str, float | None]] = []
        self.stubborn = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        assert not self.lock.locked(), "terminate must run outside the engine lock"
        self.calls.append(("terminate", None))

    def wait(self, timeout: float | None = None) -> int:
        assert not self.lock.locked(), "wait must run outside the engine lock"
        self.calls.append(("wait", timeout))
        if self.stubborn and timeout is not None:
            raise subprocess.TimeoutExpired("SYNTHETIC-engine", timeout)
        self.returncode = -9 if self.stubborn else -15
        return self.returncode

    def kill(self) -> None:
        assert not self.lock.locked(), "kill must run outside the engine lock"
        self.calls.append(("kill", None))


class TestLocalEngineStartupCleanup(unittest.TestCase):
    def setUp(self) -> None:
        # Isolate lifecycle state; model-card validation is a separate contract.
        # This backend never owns an OS process or a real checkpoint directory.
        self.backend = object.__new__(backends.LocalFx1Backend)
        self.lock = _BoundedLock()
        self.backend._engine_lock = self.lock
        self.backend._proc = None
        self.backend._verified_attach = False
        self.backend._serve_cmd = "SYNTHETIC-engine"
        self.backend._root = Path("SYNTHETIC-unused-checkpoint")
        self.backend._url = "http://engine.invalid/v1/chat/completions"
        self.backend._expected_weights_sha = "a" * 64
        self.backend._start_timeout_s = 1.0
        self.proc = _SyntheticProcess(self.lock)
        self.spawn = self.enterContext(
            patch.object(backends.subprocess, "Popen", return_value=self.proc)
        )
        self.up = self.enterContext(patch.object(self.backend, "_engine_up", return_value=False))
        self.enterContext(patch.object(backends.time, "sleep"))
        self.enterContext(patch.object(backends.time, "monotonic", return_value=0.0))
        # Any unexpected network access fails without opening a socket.
        self.urlopen = self.enterContext(
            patch.object(
                backends.urllib.request, "urlopen", side_effect=AssertionError("unexpected I/O")
            )
        )
        self.addCleanup(self.backend.close)

    def _served_weights(self, sha: str) -> None:
        self.urlopen.side_effect = None
        self.urlopen.return_value = io.BytesIO(
            json.dumps({"data": [{"fx1": {"weights_sha256": sha}}]}).encode()
        )

    def _assert_cleaned(self) -> None:
        self.assertIsNone(self.backend._proc)
        self.assertFalse(self.lock.locked())
        self.assertFalse(self.backend._verified_attach)

    def test_child_exit_reports_original_status_without_deadlock(self) -> None:
        for rc in (0, 17, -9):
            with self.subTest(returncode=rc):
                self.proc.returncode = rc
                with self.assertRaisesRegex(RuntimeError, rf"exited during startup \(rc={rc}\)"):
                    self.backend._ensure_engine()
                self._assert_cleaned()
                self.assertEqual(self.proc.calls, [])

    def test_spawned_weight_mismatch_refuses_and_reaps_outside_lock(self) -> None:
        self.up.side_effect = [False, False, True]
        self._served_weights("b" * 64)
        with self.assertRaisesRegex(
            RuntimeError, "refusing to attach to weights it did not verify"
        ):
            self.backend._ensure_engine()
        self._assert_cleaned()
        self.assertEqual(self.proc.calls, [("terminate", None), ("wait", 5)])

    def test_timeout_reaps_stubborn_child_outside_lock(self) -> None:
        self.proc.stubborn = True
        with (
            patch.object(backends.time, "monotonic", side_effect=[0.0, 0.0, 2.0]),
            self.assertRaisesRegex(RuntimeError, "did not become ready within 1s"),
        ):
            self.backend._ensure_engine()
        self._assert_cleaned()
        self.assertEqual(
            self.proc.calls,
            [("terminate", None), ("wait", 5), ("kill", None), ("wait", None)],
        )

    def test_unexpected_startup_error_reaps_child(self) -> None:
        self.up.side_effect = [False, False, OSError("SYNTHETIC probe failure")]
        with self.assertRaisesRegex(OSError, "SYNTHETIC probe failure"):
            self.backend._ensure_engine()
        self._assert_cleaned()
        self.assertEqual(self.proc.calls, [("terminate", None), ("wait", 5)])

    def test_interrupted_startup_reaps_child(self) -> None:
        self.up.side_effect = [False, False, KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            self.backend._ensure_engine()
        self._assert_cleaned()
        self.assertEqual(self.proc.calls, [("terminate", None), ("wait", 5)])

    def test_success_preserves_verified_child_until_close(self) -> None:
        self.up.side_effect = [False, False, True]
        self._served_weights("a" * 64)
        verify = self.backend._verify_served_weights

        def verify_before_publication() -> None:
            self.assertIsNone(self.backend._proc)
            verify()

        with patch.object(
            self.backend, "_verify_served_weights", side_effect=verify_before_publication
        ):
            self.backend._ensure_engine()
        self.backend._ensure_engine()
        self.spawn.assert_called_once()
        self.assertIs(self.backend._proc, self.proc)
        self.assertTrue(self.backend._verified_attach)
        self.assertEqual(self.proc.calls, [])
        self.backend.close()
        self.backend.close()
        self.assertIsNone(self.backend._proc)
        self.assertEqual(self.proc.calls, [("terminate", None), ("wait", 5)])

    def test_failed_spawn_preserves_error_without_process_cleanup(self) -> None:
        self.spawn.side_effect = OSError("SYNTHETIC missing command")
        with self.assertRaisesRegex(RuntimeError, "failed to spawn local fx-1 engine") as caught:
            self.backend._ensure_engine()
        self.assertIsInstance(caught.exception.__cause__, OSError)
        self._assert_cleaned()
        self.assertEqual(self.proc.calls, [])

    def test_attach_mismatch_does_not_stop_external_engine(self) -> None:
        self.up.return_value = True
        self._served_weights("b" * 64)
        with self.assertRaisesRegex(
            RuntimeError, "refusing to attach to weights it did not verify"
        ):
            self.backend._ensure_engine()
        self.spawn.assert_not_called()
        self._assert_cleaned()
        self.assertEqual(self.proc.calls, [])

    def test_failed_startup_can_retry_with_a_new_child(self) -> None:
        self.proc.returncode = 23
        with self.assertRaisesRegex(RuntimeError, r"\(rc=23\)"):
            self.backend._ensure_engine()
        self._assert_cleaned()
        replacement = _SyntheticProcess(self.lock)
        self.spawn.return_value = replacement
        self.up.side_effect = [False, False, True]
        self._served_weights("a" * 64)
        self.backend._ensure_engine()
        self.assertIs(self.backend._proc, replacement)
        self.assertEqual(self.spawn.call_count, 2)
        self.assertEqual(self.proc.calls, [])

    def test_failure_cleanup_does_not_detach_or_stop_a_replacement(self) -> None:
        replacement = _SyntheticProcess(self.lock)
        self.up.side_effect = [False, False, OSError("SYNTHETIC probe failure")]

        def replace_then_cleanup(proc: _SyntheticProcess) -> None:
            # Model a second caller publishing after the startup lock releases.
            self.assertFalse(self.lock.locked())
            self.backend._proc = replacement
            backends.LocalFx1Backend._close_process(proc)

        with (
            patch.object(self.backend, "_close_process", side_effect=replace_then_cleanup),
            self.assertRaisesRegex(OSError, "SYNTHETIC probe failure"),
        ):
            self.backend._ensure_engine()
        self.assertIs(self.backend._proc, replacement)
        self.assertEqual(replacement.calls, [])
        self.assertEqual(self.proc.calls, [("terminate", None), ("wait", 5)])

    def test_cleanup_failure_preserves_and_annotates_startup_refusal(self) -> None:
        self.up.side_effect = [False, False, True]
        self._served_weights("b" * 64)
        with (
            patch.object(self.proc, "terminate", side_effect=OSError("SYNTHETIC cleanup failure")),
            self.assertRaisesRegex(
                RuntimeError, "refusing to attach to weights it did not verify"
            ) as caught,
        ):
            self.backend._ensure_engine()
        self._assert_cleaned()
        self.assertIn("SYNTHETIC cleanup failure", " ".join(caught.exception.__notes__))


if __name__ == "__main__":
    unittest.main()
