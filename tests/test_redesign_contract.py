from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from crypto_timing.redesign_cache import load_redesign_cache
from crypto_timing.redesign_evaluation import _readout
from crypto_timing.redesign_model import DirectionNetwork, direction_loss
from crypto_timing.redesign_training import LABEL_MATURITY, RedesignStore


def test_distribution_readout_uses_causal_scale_and_sign_specific_magnitude() -> None:
    direction = np.zeros((2, 4))
    magnitude = np.zeros((2, 4))
    rep = np.array([[.2, .3], [.7, .8], [1.4, 1.6], [2.8, 3.2]])
    score = _readout(direction, magnitude, np.array([.01, .02]), rep)
    np.testing.assert_allclose(score, [.001, .002], atol=1e-12)
    direction[:, 2] = 20
    assert np.all(_readout(direction, magnitude, np.ones(2), rep) > score)


def test_direction_head_receives_only_realized_band_supervision() -> None:
    model = DirectionNetwork()
    inputs = (torch.randn(2, 144, 16), torch.ones(2, 144, 16),
              torch.randn(2, 72, 10), torch.ones(2, 72, 10),
              torch.randn(2, 24), torch.ones(2, 24))
    result = model(*inputs)
    result.retain_grad()
    loss = direction_loss(result, torch.tensor([0, 3]), torch.tensor([1., 0.]))
    loss.backward()
    assert torch.count_nonzero(result.grad[:, 1:3]) == 0
    assert result.grad[0, 0] != 0
    assert result.grad[1, 3] != 0


def test_v3_cache_execution_horizon_and_split_maturity() -> None:
    root = Path("outputs/redesign/cache_v3")
    if not root.exists():
        pytest.skip("built v3 cache is an optional integration fixture")
    cache = load_redesign_cache(root)
    store = RedesignStore(root)
    assert cache["manifest"]["entry_delay_minutes"] == 5
    assert cache["manifest"]["horizon_bars"] == 48
    for name, start, end in (("train", None, np.datetime64("2025-07-01")),
                             ("calibration", np.datetime64("2025-07-01"), np.datetime64("2025-11-01")),
                             ("selection", np.datetime64("2025-11-01"), np.datetime64("2026-02-01"))):
        dates = store.dates[store.splits[name] // store.n_symbols]
        if start is not None:
            assert dates.min() >= start
        assert (dates + LABEL_MATURITY).max() <= end
        assert len(store.splits[name]) % store.n_symbols == 0
