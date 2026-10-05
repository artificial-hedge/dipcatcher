import pytest


@pytest.mark.parametrize(
    "name",
    [
        "meng_po_qa_studies",
        "niutou_qa_studies",
        "wangliang_qa_studies",
        "yanwang_qa_studies",
        "heibai_qa_studies",
        "egui_qa_studies",
    ],
)
def test_w1946_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
