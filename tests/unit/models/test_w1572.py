import pytest


@pytest.mark.parametrize(
    "name",
    [
        "aster_qa_studies",
        "bluebell_qa_studies",
        "columbine_qa_studies",
        "lupine_qa_studies",
        "buttercup_qa_studies",
        "cornflower_qa_studies",
    ],
)
def test_w1572_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
