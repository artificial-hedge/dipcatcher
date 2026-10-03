from quant_fund.models.cohen_adds import bench_cohen_adds
from quant_fund.models.dense_filter import bench_dense_filter
from quant_fund.models.forcing_poset import bench_forcing_poset
from quant_fund.models.large_cardinal import bench_large_cardinal
from quant_fund.models.ma_toy import bench_ma_toy
from quant_fund.models.names_eval import bench_names_eval


def test_forcing_poset():
    assert bench_forcing_poset()["synthetic_forcing_poset"] == 1.0


def test_dense_filter():
    assert bench_dense_filter()["synthetic_dense_filter"] == 1.0


def test_names_eval():
    assert bench_names_eval()["synthetic_names_eval"] == 1.0


def test_cohen_adds():
    assert bench_cohen_adds()["synthetic_cohen_adds"] == 1.0


def test_ma_toy():
    assert bench_ma_toy()["synthetic_ma_toy"] == 1.0


def test_large_cardinal():
    assert bench_large_cardinal()["synthetic_large_cardinal"] == 1.0
