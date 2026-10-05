import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hwanung2_qa_studies",
        "dangun2_qa_studies",
        "jacheongbi2_qa_studies",
        "gamunjang2_qa_studies",
        "baridegi2_qa_studies",
        "dolhareubang2_qa_studies",
    ],
)
def test_w1822_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
