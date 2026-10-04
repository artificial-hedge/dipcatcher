import pytest


@pytest.mark.parametrize(
    "name",
    [
        "shadash_qa_studies",
        "abdir_qa_studies",
        "baal_magon_qa_studies",
        "safun_hu_qa_studies",
        "sinn_bedri_qa_studies",
        "melkob_qa_studies",
    ],
)
def test_w1876_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
