import pytest


@pytest.mark.parametrize(
    "name",
    [
        "gaia_lite_studies",
        "sqa_lite_studies",
        "simple_qa_studies",
        "tqa_lite_studies",
        "tydiqa_lite_studies",
        "coma_qa_studies",
    ],
)
def test_w1397_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
