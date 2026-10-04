from quant_fund.models.brydges_spencer import (
    bench_brydges_spencer,
)
from quant_fund.models.caracciolo_pelissetto import (
    bench_caracciolo_pelissetto,
)
from quant_fund.models.glasner_aizenman import (
    bench_glasner_aizenman,
)
from quant_fund.models.grimmett_rc import bench_grimmett_rc
from quant_fund.models.hara_hara import bench_hara_hara
from quant_fund.models.sokal_bcc import bench_sokal_bcc


def test_sokal_bcc():
    assert bench_sokal_bcc()["synthetic_sokal_bcc"] == 1.0


def test_caracciolo_pelissetto():
    assert bench_caracciolo_pelissetto()["synthetic_caracciolo_pelissetto"] == 1.0


def test_grimmett_rc():
    assert bench_grimmett_rc()["synthetic_grimmett_rc"] == 1.0


def test_hara_hara():
    assert bench_hara_hara()["synthetic_hara_hara"] == 1.0


def test_brydges_spencer():
    assert bench_brydges_spencer()["synthetic_brydges_spencer"] == 1.0


def test_glasner_aizenman():
    assert bench_glasner_aizenman()["synthetic_glasner_aizenman"] == 1.0
