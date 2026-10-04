"""Wave-933 convex-analysis canon tests."""

from __future__ import annotations

from quant_fund.models.bregman_proj import bench_bregman_proj
from quant_fund.models.conjugate_fn import bench_conjugate_fn
from quant_fund.models.fenchel_dual import bench_fenchel_dual
from quant_fund.models.moreau_env import bench_moreau_env
from quant_fund.models.proximal_map import bench_proximal_map
from quant_fund.models.subgradient_proj import bench_subgradient_proj


def test_subgradient_proj():
    assert bench_subgradient_proj()["synthetic_subgradient_proj"] == 1.0


def test_proximal_map():
    assert bench_proximal_map()["synthetic_proximal_map"] == 1.0


def test_fenchel_dual():
    assert bench_fenchel_dual()["synthetic_fenchel_dual"] == 1.0


def test_moreau_env():
    assert bench_moreau_env()["synthetic_moreau_env"] == 1.0


def test_bregman_proj():
    assert bench_bregman_proj()["synthetic_bregman_proj"] == 1.0


def test_conjugate_fn():
    assert bench_conjugate_fn()["synthetic_conjugate_fn"] == 1.0
