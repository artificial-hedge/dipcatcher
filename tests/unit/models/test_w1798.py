import pytest


@pytest.mark.parametrize(
    "name",
    [
        "nuwa2_qa_studies",
        "fuxi2_qa_studies",
        "shennong2_qa_studies",
        "changxi2_qa_studies",
        "gonggong2_qa_studies",
        "zhurong2_qa_studies",
    ],
)
def test_w1798_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
