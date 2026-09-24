"""GFDA 的标定、参数快照、worker 和像素诊断链。"""
from pathlib import Path

import numpy as np
import pytest

from app.reconstruction.engine import analyze, preview_spectrum
from app.reconstruction.diagnostics import build_pixel_analysis
from app.reconstruction.gfda_calibration import Calibration, load_calibration, save_calibration
from app.reconstruction.gfda_phase import GfdaError
from test_gfda import moving_surface


def test_calibration_roundtrip_and_configuration_gate(tmp_path):
    calibration = Calibration(np.array([1.02,1.,.98]), .05, 11., 1.5*np.pi, 'known fixture')
    path = tmp_path/'test.gfda.json'
    save_calibration(path, calibration)
    loaded = load_calibration(path, .05, 11., 1.5*np.pi)
    np.testing.assert_array_equal(calibration.coefficients, loaded.coefficients)
    with pytest.raises(GfdaError, match='匹配'):
        load_calibration(path, .04, 11., 1.5*np.pi)
    path.write_text('{"format":"other"}', encoding='utf8')
    with pytest.raises(GfdaError):
        load_calibration(path, .05, 11.)


def test_native_avc_requires_matching_sampling_and_preserves_order(tmp_path):
    path = tmp_path/'original.avc'
    path.write_text(''.join(f'{1+abs(i-10)/100}\t999\t888\n' for i in range(21)))
    loaded = load_calibration(path, .05, np.pi/(4*.05))
    assert loaded.coefficients[0] == 1.1
    json_path = tmp_path/'imported.gfda.json'
    save_calibration(json_path, loaded)
    restored = load_calibration(json_path, .05, np.pi/(4*.05))
    assert restored.phase_half_width == np.pi
    with pytest.raises(GfdaError, match='四分之一'):
        load_calibration(path, .05, 11.)


def test_gfda_saved_calibration_reproduces_result(tmp_path):
    cube, _, _ = moving_surface()
    generated = analyze(cube, .05, mode='gfda')
    path = tmp_path/'scan.gfda.json'
    save_calibration(path, generated['gfda_calibration'])
    loaded = analyze(cube, .05, mode='gfda', gfda_calibration_path=path)
    np.testing.assert_allclose(generated['h_prime'], loaded['h_prime'], atol=1e-10)
    np.testing.assert_array_equal(generated['scan_positions_used_um'], loaded['scan_positions_used_um'])


def test_gfda_pixel_diagnostics_use_formal_coordinates_and_fit():
    cube, _, _ = moving_surface()
    result = analyze(cube, .05, mode='gfda')
    payload = build_pixel_analysis(cube, 20, 5, .05, gfda_result=result)
    np.testing.assert_array_equal(result['scan_positions_used_um'], payload['signal_x'])
    assert payload['reference_origin_um'] == result['reference_origin_um'][5,20]
    delta_k = np.pi/(cube.shape[-1]*.05)
    bins = payload['fit_k_x']/delta_k
    np.testing.assert_allclose(result['fit_intercept_map'][5,20]+result['fit_slope_map'][5,20]*bins,
                               payload['fit_phase_y'], atol=1e-10)


def test_gfda_preview_is_the_corrected_formal_spectrum():
    cube, _, _ = moving_surface()
    result = analyze(cube, .05, mode='gfda')
    preview = preview_spectrum(cube, .05, analysis_method='gfda')
    assert preview['k0_value'] == pytest.approx(result['automatic_k0_value'], abs=1e-12)
    assert preview['gfda_applied']


def test_worker_runs_gfda_and_exports_scanning_coordinates(qtbot, monkeypatch, tmp_path):
    from app.gui.worker import AnalysisWorker
    from app.pipeline.session import AnalysisParams, AnalysisSession
    cube, _, _ = moving_surface()
    params = AnalysisParams(tmp_path, 0., .05, None, 3, 'weighted', 'exe', analysis_method='gfda')
    worker = AnalysisWorker(params)
    monkeypatch.setattr(worker, '_load_input_cube', lambda: cube)
    errors, finished = [], []
    worker.failed.connect(errors.append)
    worker.finished.connect(finished.append)
    worker.run()
    assert not errors
    assert len(finished) == 1
    result = finished[0]
    assert result.extras['gfda_applied']
    paths = AnalysisSession(params).export_text_results(result, tmp_path,
                selected_keys=['h_prime','scan_positions_used_um','scan_step_confidence'])
    np.testing.assert_allclose(result.extras['scan_positions_used_um'], np.loadtxt(paths['scan_positions_used_um']))


def test_invalid_pixel_remains_invalid_in_gfda_diagnostics():
    cube, _, _ = moving_surface()
    cube[0,0] = 4095
    result = analyze(cube,.05,mode='gfda')
    assert not result['valid_mask'][0,0]
    payload = build_pixel_analysis(cube,0,0,.05,gfda_result=result)
    assert not payload['fit_valid']
    assert payload['fit_point_count'] == 0
    assert np.isnan(result['h_prime'][0,0])


def test_gfda_progress_is_monotonic_and_completes():
    cube, _, _ = moving_surface()
    updates = []
    analyze(cube,.05,mode='gfda',progress=updates.append)
    assert updates[-1] == 100
    assert np.all(np.diff(updates) >= 0)
    assert len(set(updates)) > 10
