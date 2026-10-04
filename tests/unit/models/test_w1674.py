import pytest


@pytest.mark.parametrize(
    "name",
    [
        "larvae_qa_studies",
        "lemures_qa_studies",
        "manes_qa_studies",
        "penates_qa_studies",
        "lares_qa_studies",
        "genii_qa_studies",
    ],
)
def test_w1674_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
