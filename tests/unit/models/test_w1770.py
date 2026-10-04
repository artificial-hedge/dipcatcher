import pytest


@pytest.mark.parametrize(
    "name",
    [
        "nabu_qa_studies",
        "tiamat_qa_studies",
        "kingu_qa_studies",
        "etana_qa_studies",
        "gilgamesh_qa_studies",
        "enkidu_qa_studies",
    ],
)
def test_w1770_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
