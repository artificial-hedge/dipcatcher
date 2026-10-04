from quant_fund.models.branching_imm import bench_branching_imm
from quant_fund.models.crump_mode import bench_crump_mode
from quant_fund.models.galton_watson import bench_galton_watson
from quant_fund.models.kimmel_branch import bench_kimmel_branch
from quant_fund.models.multi_type_branch import (
    bench_multi_type_branch,
)
from quant_fund.models.sevastyanov import bench_sevastyanov


def test_galton_watson():
    assert bench_galton_watson()["synthetic_galton_watson"] == 1.0


def test_branching_imm():
    assert bench_branching_imm()["synthetic_branching_imm"] == 1.0


def test_multi_type_branch():
    assert bench_multi_type_branch()["synthetic_multi_type_branch"] == 1.0


def test_crump_mode():
    assert bench_crump_mode()["synthetic_crump_mode"] == 1.0


def test_kimmel_branch():
    assert bench_kimmel_branch()["synthetic_kimmel_branch"] == 1.0


def test_sevastyanov():
    assert bench_sevastyanov()["synthetic_sevastyanov"] == 1.0
