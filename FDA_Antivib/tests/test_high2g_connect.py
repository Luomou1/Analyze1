import numpy as np

from app.reconstruction.integer_regions import connection_merit, connect_regions


def test_merit_uses_periodic_eight_neighbor_differences():
    gap = np.full((5, 5), 2040, dtype=np.int64)
    gap[:, 3:] = -2040
    merit = connection_merit(gap, np.ones_like(gap, dtype=bool), 4096)
    assert 6 == merit[2, 2]  # 3*16/8，整数除法向零截断。
    assert 0 == merit[2, 1]


def test_merit_needs_three_valid_neighbors():
    valid = np.zeros((5, 5), dtype=bool)
    valid[2, 1:4] = True
    merit = connection_merit(np.zeros((5, 5), dtype=np.int64), valid, 4096)
    assert not np.isfinite(merit).any()


def test_region_growth_connects_period_crossing_with_original_threshold():
    gap = np.tile(np.arange(12) * 100 + 1800, (10, 1))
    gap = (gap + 2048) % 4096 - 2048
    result = connect_regions(gap.astype(np.int64), np.ones_like(gap, dtype=bool),
                             period=4096, noise_percent=10)
    valid = np.isfinite(result.connected)
    assert valid.all()
    np.testing.assert_array_equal(np.full((10, 11), 100), np.diff(result.connected, axis=1))
    assert 1 == len(np.unique(result.labels[valid]))


def test_regions_smaller_than_seven_are_rejected():
    valid = np.zeros((8, 8), dtype=bool)
    valid[2:4, 2:5] = True
    result = connect_regions(np.zeros((8, 8), dtype=np.int64), valid, period=4096, noise_percent=10)
    assert not np.isfinite(result.connected).any()
