import pytest


@pytest.mark.parametrize(
    "name",
    [
        "auklet_qa_studies",
        "guillemot_qa_studies",
        "frigatebird_qa_studies",
        "razorbill_qa_studies",
        "murrelet_qa_studies",
        "booby_qa_studies",
    ],
)
def test_w1498_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
