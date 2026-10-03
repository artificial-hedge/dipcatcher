from quant_fund.models.chromatic_htpy import (
    bench_chromatic_htpy,
)
from quant_fund.models.homotopy_colim import (
    bench_homotopy_colim,
)
from quant_fund.models.periodic_fam import (
    bench_periodic_fam,
)
from quant_fund.models.smash_prod import bench_smash_prod
from quant_fund.models.stable_stem2 import (
    bench_stable_stem2,
)
from quant_fund.models.unstable_tower import (
    bench_unstable_tower,
)


def test_smash_prod():
    assert bench_smash_prod()["synthetic_smash_prod"] == 1.0


def test_stable_stem2():
    assert bench_stable_stem2()["synthetic_stable_stem2"] == 1.0


def test_homotopy_colim():
    assert bench_homotopy_colim()["synthetic_homotopy_colim"] == 1.0


def test_unstable_tower():
    assert bench_unstable_tower()["synthetic_unstable_tower"] == 1.0


def test_periodic_fam():
    assert bench_periodic_fam()["synthetic_periodic_fam"] == 1.0


def test_chromatic_htpy():
    assert bench_chromatic_htpy()["synthetic_chromatic_htpy"] == 1.0
