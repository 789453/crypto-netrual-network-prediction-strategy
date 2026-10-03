from __future__ import annotations

import numpy as np

from crypto_timing.evaluation import _phase_metrics


def test_four_hour_book_charges_entry_and_terminal_exit() -> None:
    # Two consecutive 4h rebalance times, 12 contracts each.
    keys = np.concatenate((np.arange(12), np.arange(48, 60)))
    dates = np.arange(8).astype("timedelta64[h]") + np.datetime64("2025-11-01T00:00")
    funding = np.zeros((8, 12), np.float32)
    result = _phase_metrics(keys, np.ones(24), np.zeros(24), funding, dates, 12, 0,
                            side_cost=0.0006)
    phase = result["phases"][0]
    assert phase["periods"] == 2
    assert phase["active_fraction"] == 1.0
    # Entry pays 6bp once, exit pays 6bp once; held middle period has no turnover.
    assert abs(phase["cumulative_return"] - ((1 - .0006) ** 2 - 1)) < 1e-10


def test_no_trade_has_zero_cost_and_funding() -> None:
    keys = np.arange(12)
    dates = np.array([np.datetime64("2025-11-01T00:00")])
    funding = np.ones((1, 12), np.float32) * .01
    result = _phase_metrics(keys, np.zeros(12), np.ones(12), funding, dates, 12, None)
    assert result["phases"][0]["cumulative_return"] == 0


def test_linear_contract_uses_simple_price_return() -> None:
    keys = np.arange(12)
    dates = np.array([np.datetime64("2025-11-01T00:00")])
    funding = np.zeros((1, 12), np.float32)
    result = _phase_metrics(keys, np.ones(12), np.full(12, np.log(1.1)), funding,
                            dates, 12, 0, side_cost=0)
    np.testing.assert_allclose(result["phases"][0]["cumulative_return"], .1, atol=1e-12)
