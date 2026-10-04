import pytest


@pytest.mark.parametrize(
    "name",
    [
        "aramazd2_qa_studies",
        "anahit2_qa_studies",
        "vahagn2_qa_studies",
        "tir2_qa_studies",
        "astghik2_qa_studies",
        "mher2_qa_studies",
    ],
)
def test_w1833_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
