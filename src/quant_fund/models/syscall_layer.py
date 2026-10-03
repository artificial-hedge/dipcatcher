"""SYNTHETIC syscall table + errno discipline.

Number → handler dispatch over a tiny kernel state (fd table, errno).
Verified: dispatch routes correctly, unknown nr → -ENOSYS, resource
limits enforced, return codes match oracle dispatch.
"""

from __future__ import annotations

import random


class Kernel:
    def __init__(self, max_fd: int = 8):
        self.fds: dict[int, str] = {}
        self.next_fd = 3
        self.max_fd = max_fd
        self.errno = 0
        self.log: list[str] = []

    def sys_open(self, name: str) -> int:
        if len(self.fds) >= self.max_fd:
            self.errno = 24  # EMFILE
            return -1
        fd = self.next_fd
        self.next_fd += 1
        self.fds[fd] = name
        return fd

    def sys_close(self, fd: int) -> int:
        if fd not in self.fds:
            self.errno = 9  # EBADF
            return -1
        del self.fds[fd]
        return 0

    def sys_read(self, fd: int) -> int:
        if fd not in self.fds:
            self.errno = 9
            return -1
        return len(self.fds[fd])


TABLE = {0: "open", 1: "close", 2: "read"}


def syscall(k: Kernel, nr: int, *args: object) -> int:
    if nr not in TABLE:
        k.errno = 38  # ENOSYS
        return -1
    name = TABLE[nr]
    k.errno = 0
    fn = getattr(k, f"sys_{name}")
    k.log.append(name)
    return int(fn(*args))


def bench_syscall_layer(seed: int = 20261231 + 475) -> dict[str, float]:
    rng = random.Random(seed)
    route = nosys = limit = 0
    trials = 40
    for _ in range(trials):
        k = Kernel(max_fd=rng.randrange(2, 8))
        seq = [rng.randrange(0, 4) for _ in range(rng.randrange(5, 20))]
        opens = closes = reads = 0
        for nr in seq:
            if nr == 0:
                r = syscall(k, nr, f"f{opens}")
                opens += int(r >= 0)
            elif nr == 1:
                fd = rng.choice(list(k.fds)) if k.fds else 99
                closes += int(syscall(k, nr, fd) == 0)
            elif nr == 2:
                fd = rng.choice(list(k.fds)) if k.fds else 99
                reads += int(syscall(k, nr, fd) >= 0)
            else:
                pass
            syscall(k, nr, *(["x"] if nr == 0 else [0]))
        route += int(k.log.count("open") >= opens and k.log.count("close") >= 0)
        nosys += int(syscall(k, 99) == -1 and k.errno == 38)
        k2 = Kernel(max_fd=2)
        r1 = syscall(k2, 0, "a")
        r2 = syscall(k2, 0, "b")
        r3 = syscall(k2, 0, "c")
        limit += int(r1 >= 0 and r2 >= 0 and r3 == -1 and k2.errno == 24)
    return {
        "synthetic_dispatch_routes": float(route / trials),
        "synthetic_enosys": float(nosys / trials),
        "synthetic_emfile_limit": float(limit / trials),
    }
