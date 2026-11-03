"""Keep order time on the monotonic clock after an NTP step.

A forward wall jump must not flush the rate window. A backward wall jump
must not make stale data look fresh, and it must not clear a kill latch.
The kill file itself is not touched here.
"""

from __future__ import annotations

_JUMP_NS = 1_000_000_000
STALE_SECONDS = 86400.0


class ClockGuard:
    """Remember the last trusted wall sample and the monotonic anchor."""

    def __init__(self) -> None:
        self.forward = False
        self.backward = False
        self._seen = False
        self._wall = 0
        self._mono = 0
        self._anchor_wall = 0
        self._anchor_mono = 0

    def order_ns(self, wall_ns: int, mono_ns: int) -> int:
        """Order timestamp. After a jump this follows the monotonic anchor."""
        if not self._seen:
            self._remember(wall_ns, mono_ns)
            return wall_ns
        skew = (wall_ns - self._wall) - max(0, mono_ns - self._mono)
        self._note(skew, wall_ns, mono_ns)
        self._wall = wall_ns
        self._mono = mono_ns
        if not self.forward and not self.backward:
            return wall_ns
        elapsed = mono_ns - self._anchor_mono
        if elapsed < 0:
            elapsed = 0
        return self._anchor_wall + elapsed

    def _remember(self, wall_ns: int, mono_ns: int) -> None:
        self._seen = True
        self._wall = wall_ns
        self._mono = mono_ns
        self._anchor_wall = wall_ns
        self._anchor_mono = mono_ns

    def _note(self, skew: int, wall_ns: int, mono_ns: int) -> None:
        if skew >= _JUMP_NS:
            self.forward = True
            return
        if skew <= -_JUMP_NS:
            self.backward = True
            return
        if self.forward or self.backward:
            return
        self._anchor_wall = wall_ns
        self._anchor_mono = mono_ns


def freshness_seconds(now_s: float, newest_s: float, *, backward: bool) -> float:
    """Age of the newest input. A backward step fails closed at the stale SLO."""
    if backward or newest_s > now_s:
        return STALE_SECONDS
    age = now_s - newest_s
    if age < 0.0:
        return STALE_SECONDS
    return age
