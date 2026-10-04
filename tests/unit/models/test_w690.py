from quant_fund.models.homotopy_model import (
    bench_homotopy_model,
)
from quant_fund.models.homotopy_sheaf import (
    bench_homotopy_sheaf,
)
from quant_fund.models.stable_algebra import (
    bench_stable_algebra,
)
from quant_fund.models.stable_group import bench_stable_group
from quant_fund.models.stable_module import (
    bench_stable_module,
)
from quant_fund.models.stable_monoid import (
    bench_stable_monoid,
)


def test_homotopy_sheaf():
    assert bench_homotopy_sheaf()["synthetic_homotopy_sheaf"] == 1.0


def test_homotopy_model():
    assert bench_homotopy_model()["synthetic_homotopy_model"] == 1.0


def test_stable_monoid():
    assert bench_stable_monoid()["synthetic_stable_monoid"] == 1.0


def test_stable_group():
    assert bench_stable_group()["synthetic_stable_group"] == 1.0


def test_stable_module():
    assert bench_stable_module()["synthetic_stable_module"] == 1.0


def test_stable_algebra():
    assert bench_stable_algebra()["synthetic_stable_algebra"] == 1.0
