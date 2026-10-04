import pytest


@pytest.mark.parametrize(
    "name",
    [
        "sorry_bench_studies",
        "air_bench_studies",
        "salad_bench_studies",
        "wildguard_studies",
        "aegis_studies",
        "overkill_studies",
    ],
)
def test_w1311_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
