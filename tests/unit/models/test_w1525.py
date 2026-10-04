import pytest


@pytest.mark.parametrize(
    "name",
    [
        "haircap_qa_studies",
        "hornwort_qa_studies",
        "liverwort_qa_studies",
        "clubmoss_qa_studies",
        "quillwort_qa_studies",
        "sphagnum_qa_studies",
    ],
)
def test_w1525_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
