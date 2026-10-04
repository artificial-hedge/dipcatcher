from quant_fund.models.excursion_proc import (
    bench_excursion_proc,
)
from quant_fund.models.inverse_local import (
    bench_inverse_local,
)
from quant_fund.models.knight_theorem import (
    bench_knight_theorem,
)
from quant_fund.models.mazza_yor import (
    bench_mazza_yor,
)
from quant_fund.models.pitman_thm import (
    bench_pitman_thm,
)
from quant_fund.models.ray_knight import (
    bench_ray_knight,
)


def test_excursion_proc():
    assert bench_excursion_proc()["synthetic_excursion_proc"] == 1.0


def test_inverse_local():
    assert bench_inverse_local()["synthetic_inverse_local"] == 1.0


def test_ray_knight():
    assert bench_ray_knight()["synthetic_ray_knight"] == 1.0


def test_knight_theorem():
    assert bench_knight_theorem()["synthetic_knight_theorem"] == 1.0


def test_mazza_yor():
    assert bench_mazza_yor()["synthetic_mazza_yor"] == 1.0


def test_pitman_thm():
    assert bench_pitman_thm()["synthetic_pitman_thm"] == 1.0
