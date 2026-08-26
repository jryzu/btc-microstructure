"""Label-alignment and leakage tests. These are the tests that protect the
validity of the whole study."""
import numpy as np
import pandas as pd
import pytest

from src.labels import add_labels, forward_lookup


def _grid(ts, mid):
    return pd.DataFrame({"ts_ms": np.array(ts, dtype=np.int64), "mid": np.array(mid, float)})


def test_forward_return_exact_alignment():
    g = _grid([0, 500, 1000, 1500, 2000], [100, 101, 102, 103, 104])
    out = add_labels(g, horizons_s=[1], grid_ms=500)
    # at t=0, mid_{t+1s} = 102
    assert out["fwd_ret_1s"].iloc[0] == pytest.approx(np.log(102 / 100))
    # last two rows have no 1s-ahead value
    assert np.isnan(out["fwd_ret_1s"].iloc[-1])
    assert np.isnan(out["fwd_ret_1s"].iloc[-2])


def test_label_nan_on_grid_gap():
    """If the grid has a hole where t+h should be, the label must be NaN,
    never silently taken from a different timestamp."""
    g = _grid([0, 500, 2000, 2500, 3000], [100, 101, 102, 103, 104])
    out = add_labels(g, horizons_s=[1], grid_ms=500)
    # t=0: t+1000 missing (row 2 is ts=2000) -> NaN
    assert np.isnan(out["fwd_ret_1s"].iloc[0])
    # t=2000: t+1000 = 3000 exists exactly two rows ahead -> valid
    assert out["fwd_ret_1s"].iloc[2] == pytest.approx(np.log(104 / 102))


def test_no_leakage_label_uses_strictly_future_mid():
    """Injecting an artificial jump AFTER t must change the label at t but
    never any feature-visible value at t."""
    ts = list(range(0, 5000, 500))
    mid = [100.0] * len(ts)
    g = _grid(ts, mid)
    base = add_labels(g, horizons_s=[1], grid_ms=500)
    # bump mid at t=2000 (index 4) upward
    mid2 = mid.copy()
    mid2[4] = 110.0
    out = add_labels(_grid(ts, mid2), horizons_s=[1], grid_ms=500)
    # label at t=1000 (index 2) now sees the future jump
    assert out["fwd_ret_1s"].iloc[2] > base["fwd_ret_1s"].iloc[2]
    assert out["fwd_up_1s"].iloc[2] == 1.0
    # mid at t=1000 itself unchanged -> nothing feature-side moved
    assert out["mid"].iloc[2] == base["mid"].iloc[2]
    # and the label at t=2000 looks forward, not at its own bump
    assert out["fwd_ret_1s"].iloc[4] == pytest.approx(np.log(100 / 110))


def test_forward_lookup_rejects_bad_horizon():
    ts = np.array([0, 500, 1000], dtype=np.int64)
    arr = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        forward_lookup(ts, arr, 250, 500)  # not a multiple of grid


def test_directional_label_matches_return_sign():
    g = _grid([0, 500, 1000, 1500], [100, 99, 101, 100])
    out = add_labels(g, horizons_s=[1], grid_ms=500)
    assert out["fwd_up_1s"].iloc[0] == 1.0   # 100 -> 101
    assert out["fwd_up_1s"].iloc[1] == 1.0   # 99 -> 100
    valid = out["fwd_ret_1s"].notna()
    assert ((out.loc[valid, "fwd_ret_1s"] > 0) == (out.loc[valid, "fwd_up_1s"] == 1.0)).all()
