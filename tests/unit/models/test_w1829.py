import pytest


@pytest.mark.parametrize(
    "name",
    [
        "parnae2_qa_studies",
        "numgum2_qa_studies",
        "iljang2_qa_studies",
        "xiberi2_qa_studies",
        "otysi2_qa_studies",
        "nemlert2_qa_studies",
    ],
)
def test_w1829_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
