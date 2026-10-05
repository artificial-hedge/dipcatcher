import pytest


@pytest.mark.parametrize(
    "name",
    [
        "shedim_qa_studies",
        "lilim_qa_studies",
        "dybbuk_qa_studies",
        "mazzik_qa_studies",
        "seirim_qa_studies",
        "ibbur_qa_studies",
    ],
)
def test_w1915_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
