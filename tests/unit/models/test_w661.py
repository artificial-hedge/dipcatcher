from quant_fund.models.ambidexterity import bench_ambidexterity
from quant_fund.models.dieudonne_module import (
    bench_dieudonne_module,
)
from quant_fund.models.higher_semiadditivity import (
    bench_higher_semiadditivity,
)
from quant_fund.models.honda_formal import bench_honda_formal
from quant_fund.models.raynaud_height import bench_raynaud_height
from quant_fund.models.tate_height import bench_tate_height


def test_ambidexterity():
    assert bench_ambidexterity()["synthetic_ambidexterity"] == 1.0


def test_higher_semiadditivity():
    assert bench_higher_semiadditivity()["synthetic_higher_semiadditivity"] == 1.0


def test_tate_height():
    assert bench_tate_height()["synthetic_tate_height"] == 1.0


def test_dieudonne_module():
    assert bench_dieudonne_module()["synthetic_dieudonne_module"] == 1.0


def test_honda_formal():
    assert bench_honda_formal()["synthetic_honda_formal"] == 1.0


def test_raynaud_height():
    assert bench_raynaud_height()["synthetic_raynaud_height"] == 1.0
