import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hon_ma_qa_studies",
        "yeu_quai_qa_studies",
        "co_hon_qa_studies",
        "ngu_tinh_qa_studies",
        "quy_am_qa_studies",
        "tinh_linh_qa_studies",
    ],
)
def test_w1942_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
