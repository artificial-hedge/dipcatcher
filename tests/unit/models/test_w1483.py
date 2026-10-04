import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ferret_qa_studies",
        "marmot_qa_studies",
        "jackal_qa_studies",
        "moose_qa_studies",
        "raccoon_qa_studies",
        "coyote_qa_studies",
    ],
)
def test_w1483_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
