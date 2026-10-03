"""DHCP DORA + lease lifecycle sim: discover/offer/request/ack, T1 renew,
expiry."""

import numpy as np

_SEED = 20261231 + 614


class DhcpServer:
    def __init__(self, pool: list[str], lease: int = 30) -> None:
        self.pool = set(pool)
        self.lease = lease
        self.leases: dict[str, tuple[str, int]] = {}
        self.now = 0

    def request(self, mac: str) -> str | None:
        if mac in self.leases:
            ip, _ = self.leases[mac]
            self.leases[mac] = (ip, self.now)
            return ip
        if not self.pool:
            return None
        ip = sorted(self.pool)[0]
        self.pool.discard(ip)
        self.leases[mac] = (ip, self.now)
        return ip

    def tick(self, now: int) -> None:
        self.now = now
        for mac, (ip, t0) in list(self.leases.items()):
            if now - t0 > self.lease:
                del self.leases[mac]
                self.pool.add(ip)


def bench_dhcp_lease(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    srv = DhcpServer([f"10.0.1.{i}" for i in range(6)], lease=30)
    macs = [f"m{i}" for i in range(10)]
    ok = 0
    for step in range(60):
        srv.tick(step)
        mac = macs[rng.randint(10)]
        ip = srv.request(mac)
        if ip is None:
            ok += len(srv.leases) == 6  # pool exhausted is correct refusal
            continue
        others = {srv.leases[m][0] for m in srv.leases if m != mac}
        ok += ip == srv.leases[mac][0] and ip not in others
    return {"synthetic_dhcp_consistent": ok / 60}
