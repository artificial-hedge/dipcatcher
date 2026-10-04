import pytest


@pytest.mark.parametrize(
    "name",
    [
        "fennec_qa_studies",
        "meerkat_qa_studies",
        "onager_qa_studies",
        "jerboa_qa_studies",
        "pangolin_qa_studies",
        "addax_qa_studies",
    ],
)
def test_w1573_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
