import json

import numpy as np
import pytest

from app.core.surface_analysis import analyze_step, compute_step_height
from app.core.surface_options import SurfaceOptions


def test_independent_plane_does_not_remove_step_height():
    from app.core.surface_step_report import evaluate_regions
    y, x = np.indices((8, 12))
    data = 2*x + 3*y + np.where(x < 6, 0, 100.)
    measurement = compute_step_height(data, (0, 0, 6, 8), (6, 0, 12, 8))
    report = evaluate_regions(data, measurement, 'plane')
    assert -112 == report.measurement.step_height
    assert report.region_one.sq == pytest.approx(0, abs=1e-12)
    assert report.region_two.sq == pytest.approx(0, abs=1e-12)
    assert (2., 3., 0.) == pytest.approx(report.region_one.coefficients, abs=1e-12)


@pytest.mark.parametrize('reference, expected_sq', [('zero', np.sqrt(10)), ('mean', 1.)])
def test_region_reference_and_valid_fraction(reference, expected_sq):
    from app.core.surface_step_report import evaluate_regions
    data = np.array([[2., 4., 12., 14.], [np.nan, np.nan, 12., 14.]])
    measurement = compute_step_height(data, (0, 0, 2, 2), (2, 0, 4, 2))
    report = evaluate_regions(data, measurement, reference)
    assert .5 == report.region_one.valid_ratio
    assert 2 == report.region_one.valid_count
    assert expected_sq == pytest.approx(report.region_one.sq)
    assert -10 == report.measurement.step_height


def test_report_exports_matrix_and_resolved_options(tmp_path):
    from app.core.surface_step_report import evaluate_regions, export_step_report
    data = np.zeros((64, 72)); data[:, 36:] = 40; data[4, 5] = np.nan
    result = analyze_step(data, options=SurfaceOptions(remove='None', remove_spikes=False,
                          filter_type='FFT Auto', filter_mode='Low Pass'))
    measurement = compute_step_height(result.processed, (5, 5, 20, 25), (48, 5, 65, 25))
    report = evaluate_regions(result.processed, measurement, 'mean')
    path = tmp_path/'report.npz'
    export_step_report(path, result, report)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_equal(result.processed, saved['processed_nm'])
        np.testing.assert_equal(np.isfinite(result.processed), saved['valid_mask'])
        metadata = json.loads(str(saved['report_json']))
    assert result.options.high_frequency == metadata['options']['high_frequency']
    assert 'nm' == metadata['height_unit']
    assert 'mean' == metadata['regions']['reference']


def test_plane_reference_rejects_collinear_region():
    from app.core.surface_step_report import evaluate_regions
    data = np.ones((4, 8))
    measurement = compute_step_height(data, (0, 0, 4, 1), (4, 0, 8, 4))
    with pytest.raises(ValueError, match='共线|平面'):
        evaluate_regions(data, measurement, 'plane')


def test_failed_export_keeps_existing_file(tmp_path, monkeypatch):
    from app.core import surface_step_report
    result = analyze_step(np.ones((8, 10)), options=SurfaceOptions(remove='None', remove_spikes=False))
    target = tmp_path/'existing.npz'; target.write_bytes(b'original')
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(surface_step_report.np, 'savez_compressed', fail)
    with pytest.raises(OSError, match='disk full'):
        surface_step_report.export_step_report(target, result, None)
    assert b'original' == target.read_bytes()
    assert [target] == list(tmp_path.iterdir())
