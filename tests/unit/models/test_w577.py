from quant_fund.models.char_cycle import bench_char_cycle
from quant_fund.models.d_module2 import bench_d_module2
from quant_fund.models.intersection_homology import (
    bench_intersection_homology,
)
from quant_fund.models.middle_perversity import (
    bench_middle_perversity,
)
from quant_fund.models.nearby_cycles import bench_nearby_cycles
from quant_fund.models.perverse_sheaf import (
    bench_perverse_sheaf,
)


def test_perverse_sheaf():
    assert bench_perverse_sheaf()["synthetic_perverse_sheaf"] == 1.0


def test_intersection_homology():
    assert bench_intersection_homology()["synthetic_intersection_homology"] == 1.0


def test_nearby_cycles():
    assert bench_nearby_cycles()["synthetic_nearby_cycles"] == 1.0


def test_d_module2():
    assert bench_d_module2()["synthetic_d_module2"] == 1.0


def test_char_cycle():
    assert bench_char_cycle()["synthetic_char_cycle"] == 1.0


def test_middle_perversity():
    assert bench_middle_perversity()["synthetic_middle_perversity"] == 1.0
