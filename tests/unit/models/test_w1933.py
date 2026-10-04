import pytest


@pytest.mark.parametrize(
    "name",
    [
        "duppy_qa_studies",
        "soucouyant_qa_studies",
        "lagahoo_qa_studies",
        "jumbie_qa_studies",
        "ole_higue_qa_studies",
        "bacoo_qa_studies",
    ],
)
def test_w1933_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
