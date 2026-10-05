import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bdud_qa_studies",
        "srin_po_qa_studies",
        "btsan_qa_studies",
        "gdon_qa_studies",
        "gnod_sbyin_qa_studies",
        "bgegs_qa_studies",
    ],
)
def test_w1930_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
