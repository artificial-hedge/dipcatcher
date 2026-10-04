import pytest


@pytest.mark.parametrize(
    "name",
    [
        "manananggal_qa_studies",
        "minokawa_qa_studies",
        "nuno_qa_studies",
        "siyokoy_qa_studies",
        "tiyanak_qa_studies",
        "wakwak_qa_studies",
    ],
)
def test_w1659_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
