import pytest


@pytest.mark.parametrize(
    "name",
    [
        "seth_qa_studies",
        "ptah_qa_studies",
        "mut_qa_studies",
        "khepri_qa_studies",
        "atum_qa_studies",
        "amun_qa_studies",
    ],
)
def test_w1790_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
