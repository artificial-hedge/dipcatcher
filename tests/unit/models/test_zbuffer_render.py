"""Adversarial probes for zbuffer_render — nearer triangle must own overlap pixels."""

from quant_fund.models.zbuffer_render import bench_zbuffer_render, raster


def test_nearer_wins_overlap() -> None:
    far = ((5.0, 5.0, 0.8), (25.0, 5.0, 0.8), (15.0, 25.0, 0.8))
    near = ((10.0, 8.0, 0.2), (30.0, 8.0, 0.2), (20.0, 28.0, 0.2))
    owner = raster([far, near])
    cov_f = raster([far])
    cov_n = raster([near])
    assert cov_f and cov_n
    overlap = [
        (y, x) for y in range(32) for x in range(32) if cov_f[y][x] >= 0 and cov_n[y][x] >= 0
    ]
    assert overlap, "constructed triangles must overlap somewhere"
    assert all(owner[y][x] == 1 for y, x in overlap)


def test_draw_order_independent() -> None:
    far = ((5.0, 5.0, 0.8), (25.0, 5.0, 0.8), (15.0, 25.0, 0.8))
    near = ((10.0, 8.0, 0.2), (30.0, 8.0, 0.2), (20.0, 28.0, 0.2))
    a = raster([far, near])
    b = raster([near, far])
    # owner labels differ (indices swapped) but depth order same:
    # where a says 1 (near) b must say 0 (near), and vice versa
    for y in range(32):
        for x in range(32):
            if a[y][x] == 1:
                assert b[y][x] == 0
            if a[y][x] == 0:
                assert b[y][x] == 1


def test_bench_strict_passes() -> None:
    r = bench_zbuffer_render()
    assert r["synthetic_depth_order"] == 1.0
    assert r["synthetic_nearer_wins"] == 1.0


def test_bench_catches_wrong_depth(monkeypatch) -> None:
    """Inject a draw-order raster (last-wins): the bench metrics must drop below
    1.0 — the vacuous checks could not catch this."""
    import quant_fund.models.zbuffer_render as m

    real_raster = m.raster

    def broken(tris, w: int = 32, h: int = 32):
        covs = [real_raster([t]) for t in tris]
        out = [[-1] * w for _ in range(h)]
        for y in range(32):
            for x in range(32):
                for ti in range(len(tris)):
                    if covs[ti][y][x] >= 0:
                        out[y][x] = ti  # last-wins bug: ignores depth
        return out

    monkeypatch.setattr(m, "raster", broken)
    r = m.bench_zbuffer_render()
    assert r["synthetic_depth_order"] < 1.0 or r["synthetic_nearer_wins"] < 1.0
