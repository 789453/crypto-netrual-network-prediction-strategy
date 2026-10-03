from __future__ import annotations

import numpy as np
import torch

from crypto_timing.cache import _targets_for_symbol, rolling_sum
from crypto_timing.model import CryptoTimingNetwork
from crypto_timing.training import LABEL_DELAY, TRAIN_END, split_indices


def test_rolling_sum_is_trailing_and_requires_complete_window() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rolling_sum(x, 3)
    assert np.isnan(result[:2]).all()
    np.testing.assert_allclose(result[2:], [6.0, 9.0, 12.0])
    x[4] = 1000
    assert rolling_sum(x, 3)[3] == 9


def test_semivariance_uses_execution_open_and_only_future_path() -> None:
    n = 2200
    op = np.full(n, 100.0)
    cl = np.full(n, 100.0)
    r = np.zeros(n)
    r[1:] = 0.001
    end = np.array([2016])
    entry = end[0] + 2
    cl[entry] = 101.0
    cl[entry + 1 : entry + 48] = 101.0 * np.exp(np.arange(1, 48) * 0.001)
    target = _targets_for_symbol(op, cl, r, end)[0]
    assert target[0] > 0
    assert target[6] > 0
    assert target[4] > 0
    before = target.copy()
    cl[entry + 49 :] *= 100
    np.testing.assert_allclose(_targets_for_symbol(op, cl, r, end)[0], before)


def test_train_boundary_purges_unmatured_label() -> None:
    dates = np.array([TRAIN_END - LABEL_DELAY - np.timedelta64(1, "m"),
                      TRAIN_END - LABEL_DELAY + np.timedelta64(1, "m")])
    # Pad warmup hours and attach both candidate points at the end.
    dates = np.concatenate((np.full(168, np.datetime64("2023-01-01")), dates))
    targets = np.ones((len(dates), 1, 8), np.float32)
    splits = split_indices(dates, targets)
    assert 168 in splits["train"]
    assert 169 not in splits["train"]


def test_fast_tcn_has_no_future_response() -> None:
    model = CryptoTimingNetwork().eval()
    x = torch.randn(1, 288, 31)
    m = torch.ones_like(x)
    with torch.no_grad():
        projected = model.fast_project(torch.cat((x, m), dim=-1))
        original = model.fast_blocks[0](projected)
        altered = x.clone()
        altered[:, 200:] = 1000
        projected_altered = model.fast_project(torch.cat((altered, m), dim=-1))
        changed = model.fast_blocks[0](projected_altered)
    torch.testing.assert_close(original[:, :200], changed[:, :200])
