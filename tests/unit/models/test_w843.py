from quant_fund.models.b_spline import (
    bench_b_spline,
)
from quant_fund.models.blossoming import (
    bench_blossoming,
)
from quant_fund.models.box_spline import (
    bench_box_spline,
)
from quant_fund.models.cardinal_spline import (
    bench_cardinal_spline,
)
from quant_fund.models.de_boor import (
    bench_de_boor,
)
from quant_fund.models.knot_insertion import (
    bench_knot_insertion,
)


def test_b_spline():
    assert bench_b_spline()["synthetic_b_spline"] == 1.0


def test_de_boor():
    assert bench_de_boor()["synthetic_de_boor"] == 1.0


def test_cardinal_spline():
    assert bench_cardinal_spline()["synthetic_cardinal_spline"] == 1.0


def test_knot_insertion():
    assert bench_knot_insertion()["synthetic_knot_insertion"] == 1.0


def test_blossoming():
    assert bench_blossoming()["synthetic_blossoming"] == 1.0


def test_box_spline():
    assert bench_box_spline()["synthetic_box_spline"] == 1.0
