"""表面算子原厂离线回归与测量有效性边界。"""
from pathlib import Path

import numpy as np
import pytest

from app.core.surface_options import SurfaceOptions
from app.core.surface_processing import process_surface, surface_stats, _fill_holes
from app.core.surface_operators import remove_form, trim_edges, window_filter, clip_spikes

REFERENCE = Path(__file__).parent / 'data' / 'surface_native.npz'


@pytest.mark.parametrize(('name', 'mode'), [('Plane', 3), ('Sphere', 4), ('Cylinder', 6)])
def test_remove_matches_native(name, mode):
    with np.load(REFERENCE) as r:
        actual, _, _ = remove_form(r['input'], name)
        np.testing.assert_allclose(r[f'fit_{mode}'], actual, atol=1e-10)


@pytest.mark.parametrize(('mode', 'native'), [('All', 1), ('Outside', 2)])
def test_trim_matches_native(mode, native):
    with np.load(REFERENCE) as r:
        np.testing.assert_allclose(r[f'trim_{native}'], trim_edges(r['input'], 1, mode))


@pytest.mark.parametrize(('name', 'native'), [('Average', 1), ('Median', 2), ('2 Sigma', 3)])
@pytest.mark.parametrize('trim', [True, False])
def test_window_filter_matches_native(name, native, trim):
    with np.load(REFERENCE) as r:
        key = f'filter_{native}' if trim else f'filter_partial_{native}'
        np.testing.assert_allclose(r[key], window_filter(r['input'], name, 3, trim), atol=1e-11)


def test_stats_do_not_silently_recenter_and_exclude_nan():
    stats = surface_stats(np.array([[2., 4.], [np.nan, 6.]]))
    assert stats['sa'] == 4
    assert stats['sq'] == pytest.approx(np.sqrt(56/3))
    assert stats['pv'] == 4
    assert stats['valid_count'] == 3
    assert stats['valid_ratio'] == .75


def test_none_preserves_height_holes_and_input():
    z = np.array([[1., 2., 3.], [4., np.nan, 6.], [7., 8., 9.]])
    result = process_surface(z, SurfaceOptions(remove='None', remove_spikes=False))
    np.testing.assert_equal(z, result.processed)
    np.testing.assert_equal(z, result.original)


def test_three_points_reject_missing_reference():
    z = np.ones((5, 5)); z[1, 1] = np.nan
    with pytest.raises(ValueError, match='无效'):
        process_surface(z, SurfaceOptions(remove='Three Points'), ((1, 1), (1, 3), (3, 1)))


def test_no_valid_result_is_explicit_error():
    with pytest.raises(ValueError, match='有效'):
        process_surface(np.full((4, 4), np.nan), SurfaceOptions())


@pytest.mark.parametrize('key', ['fill_1_2', 'fill_2_2', 'fill_3_3'])
def test_fill_matches_native_internal_holes(key):
    with np.load(REFERENCE) as r:
        actual, _ = _fill_holes(r[key+'_input'], 25)
        np.testing.assert_allclose(r[key+'_expected'], actual, atol=1e-10)


def test_isolated_fill_matches_native():
    with np.load(REFERENCE) as r:
        actual, count = _fill_holes(r['input'], 25)
        np.testing.assert_allclose(r['fill_1'], actual, atol=1e-10)
        assert count == 1


def test_spike_matches_native():
    with np.load(REFERENCE) as r:
        actual, count = clip_spikes(r['spike_input'], 2.5)
        np.testing.assert_allclose(r['spike_expected'], actual, atol=1e-10)
        assert count == 1


def test_fill_does_not_touch_oversized_or_boundary_holes():
    z = np.ones((12, 12)); z[4:7, 4:7] = np.nan; z[0, 5] = np.nan
    actual, count = _fill_holes(z, 4)
    np.testing.assert_equal(z, actual)
    assert count == 0


@pytest.mark.parametrize(('method', 'native'), [('Average', 1), ('Median', 2), ('2 Sigma', 3)])
@pytest.mark.parametrize('avg_bad', [0, 1])
def test_filter_outliers_against_native(method, native, avg_bad):
    with np.load(REFERENCE) as r:
        actual = window_filter(r['spike_input'], method, 3, not avg_bad)
        np.testing.assert_allclose(r[f'outlier_filter_{native}_{avg_bad}'], actual, atol=1e-11)


def test_fill_cannot_make_invalid_reference_acceptable():
    data = np.ones((6, 6)); data[1, 1] = np.nan
    with pytest.raises(ValueError, match='无效'):
        process_surface(data, SurfaceOptions(remove='Three Points', data_fill=True), ((1, 1), (1, 4), (4, 1)))
