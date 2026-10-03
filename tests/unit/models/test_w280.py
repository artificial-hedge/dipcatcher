"""Wave-280 algebraic-topology module tests."""

from quant_fund.models.graph_h1 import _components
from quant_fund.models.simp_betti import _complex, betti
from quant_fund.models.winding_deg import _wrap_deg


def test_sphere_betti() -> None:
    assert betti(_complex("sphere")) == [1, 0, 1]


def test_circle_betti() -> None:
    assert betti(_complex("circle")) == [1, 1]


def test_components_tree() -> None:
    assert _components(4, [(0, 1), (1, 2), (2, 3)]) == 1


def test_winding_zero() -> None:
    assert _wrap_deg(lambda t: t * 0.0) == 0
