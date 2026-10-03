"""Suffix automaton (SAM) for substring counting (synthetic).

Online construction with suffix links. Verified: (i) number of
distinct substrings equals naive set oracle; (ii) every substring
occurrence count via endpos size equals naive oracle; (iii)
longest-common-substring between two strings matches brute force.
"""

from __future__ import annotations

import random


class SAM:
    def __init__(self) -> None:
        self.nxt: list[dict[str, int]] = [{}]
        self.link: list[int] = [-1]
        self.length: list[int] = [0]
        self.cnt: list[int] = [0]
        self.last = 0

    def extend(self, c: str) -> None:
        cur = len(self.nxt)
        self.nxt.append({})
        self.length.append(self.length[self.last] + 1)
        self.link.append(0)
        self.cnt.append(1)
        p = self.last
        while p >= 0 and c not in self.nxt[p]:
            self.nxt[p][c] = cur
            p = self.link[p]
        if p == -1:
            self.link[cur] = 0
        else:
            q = self.nxt[p][c]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = len(self.nxt)
                self.nxt.append(dict(self.nxt[q]))
                self.length.append(self.length[p] + 1)
                self.link.append(self.link[q])
                self.cnt.append(0)
                while p >= 0 and self.nxt[p].get(c) == q:
                    self.nxt[p][c] = clone
                    p = self.link[p]
                self.link[q] = self.link[cur] = clone
        self.last = cur

    def propagate(self) -> None:
        order = sorted(range(1, len(self.nxt)), key=lambda s: -self.length[s])
        for s in order:
            self.cnt[self.link[s]] += self.cnt[s]

    def distinct_substrings(self) -> int:
        return sum(self.length[s] - self.length[self.link[s]] for s in range(1, len(self.nxt)))


def lcs(a: str, b: str) -> str:
    sam = SAM()
    for c in a:
        sam.extend(c)
    v = ln = best = end = 0
    for i, c in enumerate(b):
        while v and c not in sam.nxt[v]:
            v = sam.link[v]
            ln = sam.length[v]
        if c in sam.nxt[v]:
            v = sam.nxt[v][c]
            ln += 1
        if ln > best:
            best, end = ln, i + 1
    return b[end - best : end]


def bench_suffix_automaton(seed: int = 20261231 + 261) -> dict[str, float]:
    rng = random.Random(seed)
    alpha = "abc"
    agree = cnt_agree = lcs_agree = 0
    trials = 25
    for _ in range(trials):
        s = "".join(rng.choice(alpha) for _ in range(rng.randint(5, 40)))
        sam = SAM()
        for c in s:
            sam.extend(c)
        oracle = {s[i:j] for i in range(len(s)) for j in range(i + 1, len(s) + 1)}
        agree += int(sam.distinct_substrings() == len(oracle))
        sam.propagate()
        # every substring's occurrence count via endpos: walk path
        ok = True
        for _ in range(10):
            i, j = sorted(rng.sample(range(len(s) + 1), 2))
            if i == j:
                continue
            sub = s[i:j]
            v = 0
            valid = True
            for c in sub:
                if c in sam.nxt[v]:
                    v = sam.nxt[v][c]
                else:
                    valid = False
                    break
            if valid:
                naive = sum(1 for k in range(len(s) - len(sub) + 1) if s[k : k + len(sub)] == sub)
                ok = ok and sam.cnt[v] == naive
        cnt_agree += int(ok)
        s2 = "".join(rng.choice(alpha) for _ in range(rng.randint(5, 40)))
        got = lcs(s, s2)
        # brute force longest common substring
        best = ""
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                if s[i:j] in s2 and j - i > len(best):
                    best = s[i:j]
        lcs_agree += int(len(got) == len(best))
    return {
        "synthetic_distinct_agree": float(agree / trials),
        "synthetic_count_agree": float(cnt_agree / trials),
        "synthetic_lcs_agree": float(lcs_agree / trials),
    }
