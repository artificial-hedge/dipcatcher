from quant_fund.models.boolean_prime import bench_boolean_prime
from quant_fund.models.ef_game_toy import bench_ef_game_toy
from quant_fund.models.fraisse_limit import bench_fraisse_limit
from quant_fund.models.qe_dense_order import bench_qe_dense_order
from quant_fund.models.real_closed import bench_real_closed
from quant_fund.models.vaught_test import bench_vaught_test


def test_ef_game_toy():
    assert bench_ef_game_toy()["synthetic_ef_game_toy"] == 1.0


def test_vaught_test():
    assert bench_vaught_test()["synthetic_vaught_test"] == 1.0


def test_real_closed():
    assert bench_real_closed()["synthetic_real_closed"] == 1.0


def test_boolean_prime():
    assert bench_boolean_prime()["synthetic_boolean_prime"] == 1.0


def test_fraisse_limit():
    assert bench_fraisse_limit()["synthetic_fraisse_limit"] == 1.0


def test_qe_dense_order():
    assert bench_qe_dense_order()["synthetic_qe_dense_order"] == 1.0
