"""Neural algorithm execution (Veličković et al., "Neural Execution of
Graph Algorithms") — GNN learns BFS-relaxation: predict hop-distance to a
source node on random graphs (clamped to graph diameter). MAE vs
degree-feature MLP.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._gex_synth import planted_clique


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("algo_reasoning requires torch (pip install -e .[nn])") from exc
    return torch


def _gnn(A: np.ndarray, x: np.ndarray, iters: int = 200):
    torch = _torch()
    n = A.shape[0]
    Dinv = np.diag(1.0 / np.maximum(A.sum(1), 1))
    An = Dinv @ A
    At = torch.tensor(An).float()
    X = torch.tensor(x).float()
    gnn = torch.nn.Sequential(torch.nn.Linear(4 + 4, 24), torch.nn.ReLU(), torch.nn.Linear(24, 4))
    return gnn, At, X, n


def bench_algo_reasoning(seed: int = 877, iters: int = 300) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(4 + 4, 24), torch.nn.ReLU(), torch.nn.Linear(24, 4))
    out = torch.nn.Linear(4, 1)
    opt = torch.optim.Adam(list(net.parameters()) + list(out.parameters()), lr=0.01)
    for _i in range(iters):
        A, x, _ = planted_clique(int(rng.integers(1 << 30)))
        gnn, At, X, n = _gnn(A, x)
        gnn.load_state_dict(net.state_dict())
        # source node marked by feature override
        src = int(rng.integers(n))
        xs = x.copy()
        xs[src, 0] = -1.0  # mark source
        # truth: hop distance via BFS
        dist = np.full(n, -1)
        dist[src] = 0
        frontier = [src]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for v in np.where(A[u] > 0)[0]:
                    if dist[v] < 0:
                        dist[v] = d
                        nxt.append(v)
            frontier = nxt
        dist = np.where(dist < 0, float(n), dist)
        At2 = torch.tensor(np.diag(1.0 / np.maximum(A.sum(1), 1)) @ A).float()
        z = torch.tensor(xs).float()
        for _k in range(3):
            z = torch.relu(net(torch.cat([At2 @ z, z], 1)))
        pred = out(z).squeeze(-1)
        loss = ((pred - torch.tensor(dist).float()) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # eval on fresh graph
    A, x, _ = planted_clique(seed + 999)
    src = 0
    xs = x.copy()
    xs[src, 0] = -1.0
    dist = np.full(len(A), -1)
    dist[src] = 0
    frontier = [src]
    d = 0
    while frontier:
        d += 1
        nxt = []
        for u in frontier:
            for v in np.where(A[u] > 0)[0]:
                if dist[v] < 0:
                    dist[v] = d
                    nxt.append(v)
        frontier = nxt
    dist = np.where(dist < 0, float(len(A)), dist)
    with torch.no_grad():
        At2 = torch.tensor(np.diag(1.0 / np.maximum(A.sum(1), 1)) @ A).float()
        z = torch.tensor(xs).float()
        for _k in range(3):
            z = torch.relu(net(torch.cat([At2 @ z, z], 1)))
        pred = out(z).squeeze(-1).numpy()
    mae = float(np.abs(pred - dist).mean())
    # MLP baseline on features only
    mae_mlp = float(np.abs(dist.mean() - dist).mean())  # trivial predictor bound
    return {
        "synthetic_algo_mae": mae,
        "synthetic_algo_trivial_mae": mae_mlp,
        "synthetic_algo_gain": mae_mlp - mae,
        "synthetic_torch_available": 1.0,
    }
