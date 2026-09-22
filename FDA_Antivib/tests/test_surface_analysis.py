from __future__ import annotations

import numpy as np
import pytest

import app.core.surface_analysis as surface_analysis
from app.core.surface_analysis import (
    analyze_plane,
    analyze_step,
    compute_step_height,
    load_height_matrix,
)


def test_load_height_matrix_preserves_nan(tmp_path) -> None:
    path = tmp_path / "height.txt"
    matrix = np.arange(25, dtype=float).reshape(5, 5)
    matrix[2, 2] = np.nan
    np.savetxt(path, matrix)

    loaded = load_height_matrix(path)

    assert loaded.shape == (5, 5)
    assert np.isnan(loaded[2, 2])


def test_load_height_matrix_rejects_non_matrix_input(tmp_path) -> None:
    path = tmp_path / "height.txt"
    np.savetxt(path, np.arange(8, dtype=float))

    with pytest.raises(ValueError, match="二维"):
        load_height_matrix(path)


def test_plane_analysis_preserves_original_and_removes_tilt():
    from app.core.surface_options import SurfaceOptions
    y, x = np.mgrid[:16, :20]
    data = .4*x-.3*y+12
    data[4, 5] = np.nan
    result = analyze_plane(data, options=SurfaceOptions(remove_spikes=False))
    np.testing.assert_equal(data, result.original)
    assert np.nanmax(np.abs(result.processed)) < 1e-12
    assert np.isnan(result.processed[4, 5])


def test_step_three_points_preserve_step_and_skip_holes():
    from app.core.surface_options import SurfaceOptions
    y, x = np.mgrid[:20, :30]
    data = .4*x-.3*y+12+np.where(x>15, 80., 0.)
    data[10, 24] = np.nan
    result = analyze_step(data, points=((2, 2), (15, 2), (2, 10)),
                          options=SurfaceOptions(remove='Three Points', remove_spikes=False))
    measured = compute_step_height(result.processed, (20, 5, 28, 15), (3, 5, 10, 15))
    assert measured.step_height == pytest.approx(80.)


def test_step_without_three_points_runs():
    from app.core.surface_options import SurfaceOptions
    data = np.zeros((8, 10)); data[:, 5:] = 20
    result = analyze_step(data, options=SurfaceOptions(remove='None', remove_spikes=False))
    np.testing.assert_equal(data, result.processed)
