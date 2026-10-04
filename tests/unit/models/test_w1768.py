import pytest


@pytest.mark.parametrize(
    "name",
    [
        "eshu_qa_studies",
        "nyambi_qa_studies",
        "mawu_qa_studies",
        "buluku_qa_studies",
        "chukwu_qa_studies",
        "oshumare_qa_studies",
    ],
)
def test_w1768_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
