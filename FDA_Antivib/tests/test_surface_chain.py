"""组合处理逐阶段原厂对照，原厂工件仅离线读取。"""
import json
from pathlib import Path

import numpy as np
import pytest

from app.core.surface_options import SurfaceOptions
from app.core.surface_processing import process_surface, _fill_holes
from app.core.surface_operators import trim_edges, clip_spikes, window_filter, remove_form

FIXTURE = Path(__file__).parent/'data'/'surface_chain_native.npz'
CASES = json.loads(FIXTURE.with_suffix('.npz.json').read_text(encoding='utf-8'))['cases']


@pytest.mark.parametrize('case', CASES)
def test_native_chain_stage_matrices_and_masks(case):
    index = case['index']
    with np.load(FIXTURE) as native:
        working = native[f'input_{index}']
        for stage in case['stages']:
            match stage:
                case 'trim':
                    working = trim_edges(working, case['trim'], case['trim_mode'])
                case 'spikes':
                    working = clip_spikes(working, 2.5)[0]
                case 'filter':
                    working = window_filter(working, 'Average', 3, False)
                case 'form':
                    working = remove_form(working, 'Plane')[0]
                case 'fill':
                    working = _fill_holes(working, 25)[0]
            np.testing.assert_allclose(native[f'{stage}_{index}'], working, atol=1e-9, rtol=0)
        result = process_surface(native[f'input_{index}'], SurfaceOptions(
            trim=case['trim'], trim_mode=case['trim_mode'], filter_mode=case['filter_mode'], data_fill=True,
            data_fill_method='Polynomial'))
        np.testing.assert_allclose(native[f'fill_{index}'], result.processed, atol=1e-9, rtol=0)


def test_large_sparse_surface_is_repeatable_and_preserves_input():
    y, x = np.indices((1024, 1280), dtype=float)
    data = .01*x-.02*y+np.sin(x/9)
    data[::3, ::3] = np.nan
    original = data.copy()
    options = SurfaceOptions(remove='Plane', remove_spikes=False, data_fill=True)
    first = process_surface(data, options)
    second = process_surface(data, options)
    np.testing.assert_equal(first.processed, second.processed)
    np.testing.assert_equal(original, data)
    assert first.filled_count > 100000


@pytest.mark.parametrize('filter_type', ['FFT Fixed', 'Gauss Spline', 'Robust Gauss Spline'])
def test_large_filter_nan_domain_and_equivalent_physical_scale(filter_type):
    from dataclasses import replace
    y, x = np.indices((256, 384), dtype=float)
    data = 20+np.sin(x/8)+.2*np.cos(y/5)
    data[::4, ::5] = np.nan
    options = SurfaceOptions(remove='None', remove_spikes=False, filter_mode='Low Pass',
                             filter_type=filter_type, pixel_size_um=.48, high_frequency=150)
    first = process_surface(data, options)
    second = process_surface(data, replace(options, pixel_size_um=4.8e-7, high_frequency=1.5e8))
    np.testing.assert_equal(np.isfinite(data), np.isfinite(first.processed))
    np.testing.assert_allclose(first.processed, second.processed, atol=1e-9)


def test_trim_and_minimum_area_precede_fill():
    data = np.full((14, 14), np.nan)
    data[1:6, 1:6] = 1
    data[8:13, 8:13] = 10
    data[3, 3] = np.nan
    result = process_surface(data, SurfaceOptions(remove='None', remove_spikes=False, trim=1,
                             min_area=6, data_fill=True, data_fill_max=25))
    assert not np.isfinite(result.processed[:7, :7]).any()
    assert 9 == np.isfinite(result.processed).sum()
