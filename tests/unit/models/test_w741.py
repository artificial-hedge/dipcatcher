from quant_fund.models.aiten_chayes import bench_aiten_chayes
from quant_fund.models.beffara_nolin import bench_beffara_nolin
from quant_fund.models.gandre_liggett import bench_gandre_liggett
from quant_fund.models.hara_slade import bench_hara_slade
from quant_fund.models.heyman_redner import bench_heyman_redner
from quant_fund.models.newman_percolation import (
    bench_newman_percolation,
)


def test_beffara_nolin():
    assert bench_beffara_nolin()["synthetic_beffara_nolin"] == 1.0


def test_hara_slade():
    assert bench_hara_slade()["synthetic_hara_slade"] == 1.0


def test_gandre_liggett():
    assert bench_gandre_liggett()["synthetic_gandre_liggett"] == 1.0


def test_heyman_redner():
    assert bench_heyman_redner()["synthetic_heyman_redner"] == 1.0


def test_aiten_chayes():
    assert bench_aiten_chayes()["synthetic_aiten_chayes"] == 1.0


def test_newman_percolation():
    assert bench_newman_percolation()["synthetic_newman_percolation"] == 1.0
