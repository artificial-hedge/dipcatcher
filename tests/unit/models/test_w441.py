from quant_fund.models.filtered_module import bench_filtered_module
from quant_fund.models.fontaine_ring import bench_fontaine_ring
from quant_fund.models.gal_rep import bench_gal_rep
from quant_fund.models.hecke_eigensys import bench_hecke_eigensys
from quant_fund.models.ribet_toy import bench_ribet_toy
from quant_fund.models.weil_deligne import bench_weil_deligne


def test_gal_rep():
    assert bench_gal_rep()["synthetic_gal_rep"] == 1.0


def test_fontaine_ring():
    assert bench_fontaine_ring()["synthetic_fontaine_ring"] == 1.0


def test_filtered_module():
    assert bench_filtered_module()["synthetic_filtered_module"] == 1.0


def test_weil_deligne():
    assert bench_weil_deligne()["synthetic_weil_deligne"] == 1.0


def test_hecke_eigensys():
    assert bench_hecke_eigensys()["synthetic_hecke_eigensys"] == 1.0


def test_ribet_toy():
    assert bench_ribet_toy()["synthetic_ribet_toy"] == 1.0
