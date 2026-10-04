import pytest


@pytest.mark.parametrize(
    "name",
    [
        "awq_studies",
        "entropy_code_quant_studies",
        "gptq_studies",
        "kv_cache_quant_studies",
        "smoothquant_studies",
        "weight_share_studies",
    ],
)
def test_w1290_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
