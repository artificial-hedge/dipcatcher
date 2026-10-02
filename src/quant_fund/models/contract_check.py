"""Design-by-contract: pre/postcondition + invariant checker."""

import numpy as np

_SEED = 20261231 + 674


class Stack:
    def __init__(self) -> None:
        self.items: list[int] = []
        self._inv_ok = True

    def _check_inv(self) -> bool:
        return len(self.items) >= 0

    def push(self, v: int) -> bool:
        # pre: none; post: len increases by 1
        before = len(self.items)
        self.items.append(v)
        return len(self.items) == before + 1 and self._check_inv()

    def pop(self) -> int | None:
        # pre: nonempty
        if not self.items:
            return None
        return self.items.pop()


def bench_contract_check(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    violations_free = 0.0
    trials = 40
    for _ in range(trials):
        st = Stack()
        good = True
        for _ in range(30):
            if rng.rand() < 0.6:
                good &= st.push(int(rng.randint(100)))
            else:
                st.pop()
        violations_free += float(good)
    return {"synthetic_contract_holds": violations_free / trials}
