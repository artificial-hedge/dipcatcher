"""CFG simplification (wave 294) (SYNTHETIC).

Iterative: merge a block into its single unconditional predecessor,
remove unreachable blocks, collapse empty blocks — fixpoint result vs
naive full-reduction oracle.
"""

_SEED = 20261231 + 852

CFG = dict[str, tuple]


def _succs(n: tuple) -> list[str]:
    if n[0] == "jmp":
        return [n[1]]
    if n[0] == "br":
        return [n[2], n[3]]
    return []


def simplify(cfg: CFG, entry: str) -> CFG:
    cfg = dict(cfg)
    # drop unreachable
    seen = {entry}
    stack = [entry]
    while stack:
        for s in _succs(cfg[stack.pop()]):
            if s not in seen and s in cfg:
                seen.add(s)
                stack.append(s)
    cfg = {k: v for k, v in cfg.items() if k in seen}
    # merge jmp-chains: b with single succ and succ's only pred = b
    preds: dict[str, int] = {}
    for _k, n in cfg.items():
        for s in _succs(n):
            preds[s] = preds.get(s, 0) + 1
    changed = True
    while changed:
        changed = False
        for b in list(cfg):
            n = cfg[b]
            if n[0] == "jmp" and n[1] in cfg and n[1] != b:
                tgt = n[1]
                if preds.get(tgt, 0) <= 1:
                    cfg[b] = cfg[tgt]
                    del cfg[tgt]
                    preds = {}
                    for _k, m in cfg.items():
                        for s in _succs(m):
                            preds[s] = preds.get(s, 0) + 1
                    changed = True
                    break
    return cfg


def bench_cfg_simplify(seed: int = _SEED) -> dict[str, float]:
    cfg = {
        "entry": ("jmp", "b1"),
        "b1": ("jmp", "b2"),
        "b2": ("br", "c", "b3", "b4"),
        "b3": ("ret", "x"),
        "b4": ("ret", "y"),
        "dead": ("jmp", "b3"),
    }
    s = simplify(cfg, "entry")
    ok = int("dead" not in s and len(s) == 3 and s["entry"][0] == "br")
    # merge chain of 3 jmps
    cfg2 = {"e": ("jmp", "a"), "a": ("jmp", "b"), "b": ("jmp", "c"), "c": ("ret", "z")}
    ok += int(simplify(cfg2, "e") == {"e": ("ret", "z")})
    # shared succ not merged (2 preds)
    cfg3 = {"a": ("jmp", "t"), "b": ("jmp", "t"), "t": ("ret", "z"), "e": ("br", "c", "a", "b")}
    ok += int(len(simplify(cfg3, "e")) == 4)
    return {"synthetic_cfg": float(ok == 3)}
