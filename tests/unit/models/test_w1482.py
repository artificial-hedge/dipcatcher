import pytest


@pytest.mark.parametrize(
    "name",
    [
        "marten_qa_studies",
        "ermine_qa_studies",
        "fisher_qa_studies",
        "mink_qa_studies",
        "polecat_qa_studies",
        "wolverine_qa_studies",
    ],
)
def test_w1482_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
