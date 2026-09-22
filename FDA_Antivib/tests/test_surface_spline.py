"""普通样条的原厂矩阵对照与参数边界。"""
from pathlib import Path

import numpy as np
import pytest

from app.core.surface_options import SurfaceOptions
from app.core.surface_processing import process_surface
from app.gui.surface_analysis_page import PlaneAnalysisPage

REFERENCE = Path(__file__).parent / 'data' / 'surface_spline_native.npz'


@pytest.mark.parametrize('index', range(8))
@pytest.mark.parametrize('mode', ['Low Pass', 'High Pass', 'Band Pass'])
@pytest.mark.parametrize('robust', [False, True])
def test_gauss_spline_matches_native(index: int, mode: str, robust: bool) -> None:
    path = REFERENCE.with_name('surface_robust_spline_native.npz') if robust else REFERENCE
    with np.load(path) as reference:
        mode_index = ('Low Pass', 'High Pass', 'Band Pass').index(mode)
        result = process_surface(reference[f'input_{index}'], SurfaceOptions(
            remove='None', remove_spikes=False,
            filter_type='Robust Gauss Spline' if robust else 'Gauss Spline', filter_mode=mode,
            pixel_size_um=10, low_frequency=float(reference[f'floor_{index}'])*100,
            high_frequency=float(reference[f'ceiling_{index}'])*100))
        np.testing.assert_allclose(reference[f'expected_{index}_{mode_index}'], result.processed, atol=2e-11, rtol=0)


@pytest.mark.parametrize('algorithm', ['Gauss Spline', 'Robust Gauss Spline'])
def test_spline_page_worker_uses_selected_algorithm(qtbot, tmp_path: Path, algorithm: str) -> None:
    reference_path = REFERENCE.with_name('surface_robust_spline_native.npz') if algorithm.startswith('Robust') else REFERENCE
    with np.load(reference_path) as reference:
        data, expected = reference['input_1'], reference['expected_1_2']
    path = tmp_path / 'spline.txt'
    np.savetxt(path, data)
    page = PlaneAnalysisPage()
    qtbot.addWidget(page)
    controls = page.surface_controls
    controls.remove.setCurrentIndex(controls.remove.findData('None'))
    controls.spikes.setChecked(False)
    controls.filter_mode.setCurrentText('Band Pass')
    controls.filter_type.setCurrentText(algorithm)
    controls.pixel_size.setValue(10)
    controls.low_frequency.setValue(10)
    controls.high_frequency.setValue(20)
    page.file_edit.setText(str(path))
    page._run_analysis()
    qtbot.waitUntil(lambda: page._result is not None and page._analysis_thread is None, timeout=8000)
    np.testing.assert_allclose(expected, page._result.processed, atol=2e-11, rtol=0)


@pytest.mark.parametrize('algorithm', ['Gauss Spline', 'Robust Gauss Spline'])
def test_spline_rejects_invalid_cutoff_and_shape(algorithm: str) -> None:
    with pytest.raises(ValueError):
        SurfaceOptions(filter_type=algorithm, filter_mode='Band Reject', pixel_size_um=10,
                       low_frequency=20, high_frequency=10)
    with pytest.raises(ValueError):
        SurfaceOptions(filter_type=algorithm, filter_mode='Low Pass', pixel_size_um=10,
                       high_frequency=20, cutoff_shape='Unknown')
