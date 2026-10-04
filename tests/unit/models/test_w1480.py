import pytest


@pytest.mark.parametrize(
    "name",
    [
        "mamba_qa_studies",
        "taipan_qa_studies",
        "krait_qa_studies",
        "boa_qa_studies",
        "monitor_qa_studies",
        "adder_qa_studies",
    ],
)
def test_w1480_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
