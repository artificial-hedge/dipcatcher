from quant_fund.models.donaldson_thomas import bench_donaldson_thomas
from quant_fund.models.gopakumar_vafa import bench_gopakumar_vafa
from quant_fund.models.gw_descendant import bench_gw_descendant
from quant_fund.models.kontsevich_mgn import bench_kontsevich_mgn
from quant_fund.models.mnop_conj import bench_mnop_conj
from quant_fund.models.pandharipande_thomas import (
    bench_pandharipande_thomas,
)


def test_kontsevich_mgn():
    assert bench_kontsevich_mgn()["synthetic_kontsevich_mgn"] == 1.0


def test_gw_descendant():
    assert bench_gw_descendant()["synthetic_gw_descendant"] == 1.0


def test_donaldson_thomas():
    assert bench_donaldson_thomas()["synthetic_donaldson_thomas"] == 1.0


def test_pandharipande_thomas():
    assert bench_pandharipande_thomas()["synthetic_pandharipande_thomas"] == 1.0


def test_gopakumar_vafa():
    assert bench_gopakumar_vafa()["synthetic_gopakumar_vafa"] == 1.0


def test_mnop_conj():
    assert bench_mnop_conj()["synthetic_mnop_conj"] == 1.0
