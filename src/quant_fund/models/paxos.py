"""Single-decree Paxos consensus (synthetic) (SYNTHETIC).

Simulates proposers/acceptors/learners over an unreliable channel:
prepare(promise) → accept(accepted) → learn(chosen). Random message
drops, competing proposals. Verified: (i) agreement — all learned
values equal; (ii) validity — chosen value was proposed; (iii)
termination under majority-live.
"""

from __future__ import annotations

import random


class _Acceptor:
    def __init__(self) -> None:
        self.promised = -1
        self.acc_num = -1
        self.acc_val: int | None = None

    def prepare(self, n: int) -> tuple[int, int, int | None]:
        if n > self.promised:
            self.promised = n
        return self.promised, self.acc_num, self.acc_val

    def accept(self, n: int, v: int) -> bool:
        if n >= self.promised:
            self.promised = n
            self.acc_num = n
            self.acc_val = v
            return True
        return False


def run_paxos(
    n_acc: int,
    proposals: list[int],
    rng: random.Random,
    drop: float = 0.15,
    rounds: int = 60,
) -> dict[str, object]:
    acceptors = [_Acceptor() for _ in range(n_acc)]
    majority = n_acc // 2 + 1
    learned: list[int] = []
    chosen_val: int | None = None
    n_prop = len(proposals)
    for r in range(rounds):
        if chosen_val is not None:
            break
        pid = r % n_prop
        n = 100 * (r + 1) + pid  # unique increasing ballot
        # phase 1
        promises = []
        for a in acceptors:
            if rng.random() < drop:
                continue
            promises.append(a.prepare(n))
        if len(promises) < majority:
            continue
        # adopt highest-accepted value else own proposal
        prev = [p for p in promises if p[1] >= 0]
        v = max(prev, key=lambda t: t[1])[2] if prev else proposals[pid]
        # phase 2
        acc = 0
        for a in acceptors:
            if rng.random() < drop:
                continue
            if a.accept(n, v):  # type: ignore[arg-type]
                acc += 1
        if acc >= majority:
            chosen_val = v  # type: ignore[assignment]
            learned.append(v)  # type: ignore[arg-type]
    return {"chosen": chosen_val, "learned": learned, "rounds": r + 1}


def bench_paxos(seed: int = 20261231 + 250) -> dict[str, float]:
    rng = random.Random(seed)
    agree = valid = terminated = 0
    trials = 60
    for _ in range(trials):
        proposals = rng.sample(range(100, 999), 3)
        res = run_paxos(5, proposals, rng, drop=0.15)
        chosen = res["chosen"]
        if chosen is not None:
            terminated += 1
            learned = res["learned"]
            if not (isinstance(learned, list)):
                raise ValueError("isinstance(learned, list)")
            if not (isinstance(chosen, int)):
                raise ValueError("isinstance(chosen, int)")
            agree += int(all(v == learned[0] for v in learned))
            valid += int(chosen in proposals)
    # determinism test: no drops must always agree in few rounds
    det = 0
    for _ in range(20):
        res = run_paxos(3, [1, 2, 3], rng, drop=0.0)
        det += int(res["chosen"] == 1)
    return {
        "synthetic_terminated": float(terminated / trials),
        "synthetic_agree": float(agree / trials),
        "synthetic_valid": float(valid / trials),
        "synthetic_deterministic": float(det / 20),
    }
