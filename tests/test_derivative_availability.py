from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_timing.derivative_cache import _asof_derivative_features


def test_mark_bar_and_funding_require_their_publication_buffer(tmp_path) -> None:
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    date = pd.date_range(start, periods=3, freq="h")
    frame = pd.DataFrame({
        "date": date,
        "mark_close": [101.0, 102.0, 103.0],
        "index_close": [100.0, 100.0, 100.0],
        "mark_close_time": date + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        "index_close_time": date + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        "funding_rate": [0.001, np.nan, 0.005],
    })
    path = tmp_path / "derivatives.parquet"
    frame.to_parquet(path)
    decisions = np.array(["2025-01-01T01:00", "2025-01-01T02:00",
                          "2025-01-01T02:05", "2025-01-01T03:00"], dtype="datetime64[m]")
    result = _asof_derivative_features(path, decisions)
    assert np.isnan(result[0, 0])
    np.testing.assert_allclose(result[1, 0], np.log(101 / 100), atol=1e-7)
    np.testing.assert_allclose(result[2, 0], np.log(102 / 100), atol=1e-7)
    np.testing.assert_allclose(result[1, 4], .001, atol=1e-8)
    np.testing.assert_allclose(result[3, 4], .005, atol=1e-8)
    frame.loc[2, ["mark_close", "index_close", "funding_rate"]] = [999, 999, 999]
    frame.to_parquet(path)
    changed = _asof_derivative_features(path, decisions)
    np.testing.assert_allclose(result[:2], changed[:2], equal_nan=True)
    np.testing.assert_allclose(result[2, :4], changed[2, :4], equal_nan=True)
    assert changed[2, 4] > result[2, 4]
