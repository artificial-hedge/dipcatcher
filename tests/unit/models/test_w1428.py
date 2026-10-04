import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bureau_qa_studies",
        "election_qa_studies",
        "cabinet_qa_studies",
        "government_qa_studies",
        "ministry_qa_studies",
        "agency_qa_studies",
    ],
)
def test_w1428_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
