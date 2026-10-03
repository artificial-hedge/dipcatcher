"""Unit tests for wave-291 VLSI/EDA canon modules."""

from quant_fund.models.a_star_route import route
from quant_fund.models.drc_check import spacing_violation, width_violation
from quant_fund.models.levelize import levelize
from quant_fund.models.netlist_parse import NETLIST, parse, simulate
from quant_fund.models.place_quadratic import place
from quant_fund.models.sta_timing import longest_path


def test_netlist_adder():
    inputs, gates, outs = parse(NETLIST)
    assert len(gates) == 5
    r = simulate(inputs, gates, outs, {"a0": 1, "b0": 1, "cin": 1})
    assert r["s"] == 1 and r["cout"] == 1


def test_sta_diamond():
    at = longest_path({0: [1, 2], 1: [3], 2: [3]}, {0: 0.5, 1: 3.0, 2: 1.0, 3: 0.0}, [0], 4)
    assert abs(at[3] - 3.5) < 1e-9


def test_route_wall():
    grid = [[0] * 8 for _ in range(8)]
    for i in range(1, 7):
        grid[i][4] = 1
    assert route(grid, (0, 0), (7, 7)) == 14


def test_drc():
    assert spacing_violation([(0, 0, 1, 1), (1.5, 0, 2.5, 1)], 0.6) == 1
    assert width_violation([(0, 0, 0.5, 2)], 0.6) == 1


def test_place_centroid():
    pos = place(
        1,
        {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.0, 1.0), 4: (1.0, 1.0)},
        [[0, 1], [0, 2], [0, 3], [0, 4]],
    )
    assert abs(pos[0][0] - 0.5) < 1e-6


def test_levelize_tree():
    lvl = levelize(["i0", "i1"], {"g": ("AND", ["i0", "i1", "n1"])})
    assert lvl["n1"] == 1
