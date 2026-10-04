import pytest


@pytest.mark.parametrize(
    "name",
    [
        "quail_dove_qa_studies",
        "ground_dove_qa_studies",
        "fruit_dove_qa_studies",
        "cuckoo_dove_qa_studies",
        "emerald_dove_qa_studies",
        "crowned_pigeon_qa_studies",
    ],
)
def test_w1547_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
