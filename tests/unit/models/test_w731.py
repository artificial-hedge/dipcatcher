from quant_fund.models.brylinski_kato import (
    bench_brylinski_kato,
)
from quant_fund.models.higher_ramif import bench_higher_ramif
from quant_fund.models.log_ramification import (
    bench_log_ramification,
)
from quant_fund.models.neron_raynaud import (
    bench_neron_raynaud,
)
from quant_fund.models.semi_stable_model import (
    bench_semi_stable_model,
)
from quant_fund.models.temkin_alter import (
    bench_temkin_alter,
)


def test_higher_ramif():
    assert bench_higher_ramif()["synthetic_higher_ramif"] == 1.0


def test_brylinski_kato():
    assert bench_brylinski_kato()["synthetic_brylinski_kato"] == 1.0


def test_log_ramification():
    assert bench_log_ramification()["synthetic_log_ramification"] == 1.0


def test_semi_stable_model():
    assert bench_semi_stable_model()["synthetic_semi_stable_model"] == 1.0


def test_neron_raynaud():
    assert bench_neron_raynaud()["synthetic_neron_raynaud"] == 1.0


def test_temkin_alter():
    assert bench_temkin_alter()["synthetic_temkin_alter"] == 1.0
