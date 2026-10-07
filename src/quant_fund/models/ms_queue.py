"""Michael-Scott lock-free queue simulator + linearizability oracle (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 582


class MSQueue:
    """Linked list with lagging tail; CAS semantics modelled atomically."""

    def __init__(self) -> None:
        self.head: dict | None = {"val": None, "next": None}
        self.tail = self.head

    def enq(self, v: int) -> None:
        node = {"val": v, "next": None}
        while True:
            t = self.tail
            nxt = t["next"]
            if nxt is None:
                # CAS(t.next, None, node)
                t["next"] = node
                # CAS(tail, t, node) may lag — apply lazily
                if self.tail is t:
                    self.tail = node
                return
            # help swing tail forward
            self.tail = nxt

    def deq(self) -> int | None:
        while True:
            h = self.head
            t = self.tail
            if h is None:
                return None
            nxt = h["next"]
            if h is t:
                if nxt is None:
                    return None
                self.tail = nxt  # help
                continue
            if nxt is not None:
                v = int(nxt["val"])
                self.head = nxt  # CAS(head)
                return v


def _linearizability_oracle(ops: list[tuple[str, int]]) -> bool:
    seq: list[int] = []
    for op, v in ops:
        if op == "enq":
            seq.append(v)
        else:
            if v == -1:
                if seq:
                    return False
            else:
                if not seq or seq.pop(0) != v:
                    return False
    return True


def bench_ms_queue(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(30):
        q = MSQueue()
        record: list[tuple[str, int]] = []
        counter = 0
        for _ in range(120):
            if rng.rand() < 0.55:
                counter += 1
                q.enq(counter)
                record.append(("enq", counter))
            else:
                v = q.deq()
                record.append(("deq", v if v is not None else -1))
        if _linearizability_oracle(record):
            ok += 1
    return {"synthetic_msq_linearizable": ok / 30}
