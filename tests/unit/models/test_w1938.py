import pytest


@pytest.mark.parametrize(
    "name",
    [
        "phi_pha_qa_studies",
        "nang_mai_qa_studies",
        "krahang_qa_studies",
        "krasue_qa_studies",
        "phi_am_qa_studies",
        "phi_hong_qa_studies",
    ],
)
def test_w1938_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
