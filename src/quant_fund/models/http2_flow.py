"""SYNTHETIC HTTP/2 flow control — per-stream + connection windows.

Sender can only put min(stream_win, conn_win) bytes in flight;
WINDOW_UPDATE restores credit. Verified: in-flight never exceeds
windows, WINDOW_UPDATE allows resumed progress.
"""

from __future__ import annotations

import random


class Conn:
    def __init__(self, win: int):
        self.win = win
        self.streams: dict[int, int] = {}
        self.in_flight_conn = 0
        self.in_flight_stream: dict[int, int] = {}

    def open_stream(self, sid: int, win: int) -> None:
        self.streams[sid] = win
        self.in_flight_stream[sid] = 0

    def send(self, sid: int, n: int) -> int:
        allow = min(self.win - self.in_flight_conn, self.streams[sid] - self.in_flight_stream[sid])
        n = min(n, max(0, allow))
        self.in_flight_conn += n
        self.in_flight_stream[sid] += n
        return n

    def ack_conn(self, n: int) -> None:
        self.in_flight_conn = max(0, self.in_flight_conn - n)

    def window_update(self, sid: int, inc: int) -> None:
        self.streams[sid] += inc
        self.in_flight_stream[sid] = max(0, self.in_flight_stream[sid] - inc)


def bench_http2_flow(seed: int = 20261231 + 405) -> dict[str, float]:
    _ = random.Random(seed)
    bound = resume = conn_ok = 0
    trials = 40
    for _ in range(trials):
        c = Conn(100)
        c.open_stream(1, 60)
        c.open_stream(3, 60)
        ok = True
        # stream-1 can only put 60 in flight; conn caps at 100
        s1 = c.send(1, 100)
        ok &= s1 == 60
        s2 = c.send(3, 100)
        ok &= s2 == 40  # conn left: 40
        bound += int(ok)
        # WINDOW_UPDATE on stream 1 frees stream credit but not conn
        c.window_update(1, 30)
        s3 = c.send(1, 50)
        resume += int(s3 == 0)  # conn exhausted at 100
        c.ack_conn(50)
        s4 = c.send(1, 50)
        resume += int(s4 == 50)  # min(conn 50, stream 60)
        # conn-level bound: total in flight ≤ 100
        conn_ok += int(c.in_flight_conn <= 100)
    return {
        "synthetic_window_bounded": float(bound / trials),
        "synthetic_update_resumes": float(resume / (trials * 2)),
        "synthetic_conn_cap": float(conn_ok / trials),
    }
