"""FFT Gaussian 原厂回归、单位换算与参数边界。"""
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from app.core.surface_options import SurfaceAnalysisError, SurfaceOptions
from app.core.surface_processing import process_surface
from app.gui.surface_controls import SurfaceControls
from app.gui.surface_analysis_page import PlaneAnalysisPage

REFERENCE = Path(__file__).parent / 'data' / 'surface_fft_native.npz'
MODES = ('Low Pass', 'High Pass', 'Band Pass', 'Band Reject')


@pytest.mark.parametrize('index', range(8))
@pytest.mark.parametrize('mode', range(4))
@pytest.mark.parametrize('cutoff', ['Gaussian', 'Sinusoid'])
def test_fft_fixed_matches_native(index: int, mode: int, cutoff: str) -> None:
    path = REFERENCE if cutoff == 'Gaussian' else REFERENCE.with_name('surface_sinusoid_native.npz')
    with np.load(path) as reference:
        options = SurfaceOptions(remove='None', remove_spikes=False, filter_type='FFT Fixed',
                                 filter_mode=MODES[mode], low_frequency=float(reference[f'floor_{index}'])*100,
                                 high_frequency=float(reference[f'ceiling_{index}'])*100,
                                 pixel_size_um=10, cutoff_shape=cutoff)
        result = process_surface(reference[f'input_{index}'], options)
        np.testing.assert_allclose(reference[f'expected_{index}_{mode}'], result.processed, atol=2e-11, rtol=0)
        np.testing.assert_equal(reference[f'input_{index}'], result.original)
        assert result.options == options


@pytest.mark.parametrize('changes', [
    {'pixel_size_um': 0}, {'pixel_size_um': float('nan')},
    {'high_frequency': 0}, {'high_frequency': float('inf')},
    {'cutoff_shape': 'Unknown'},
    {'filter_mode': 'Band Pass', 'low_frequency': 30},
])
def test_fft_rejects_unusable_or_unverified_parameters(changes: dict) -> None:
    values = dict(filter_type='FFT Fixed', filter_mode='Low Pass', pixel_size_um=10,
                  low_frequency=10, high_frequency=20)
    values.update(changes)
    with pytest.raises(SurfaceAnalysisError):
        SurfaceOptions(**values)


def test_frequency_and_wavelength_controls_reciprocate(qtbot) -> None:
    controls = SurfaceControls()
    qtbot.addWidget(controls)
    controls.filter_mode.setCurrentText('Low Pass')
    controls.filter_type.setCurrentText('FFT Fixed')
    controls.pixel_size.setValue(10)
    controls.high_frequency.setValue(20)
    assert controls.high_wavelength.value() == pytest.approx(.05)
    controls.high_wavelength.setValue(.025)
    assert controls.high_frequency.value() == pytest.approx(40)
    assert controls.options().high_frequency == 40
    assert controls.options().pixel_size_um == 10
    assert not controls.low_frequency.isEnabled()
    assert not controls.window.isEnabled()
    assert controls.filter_trim.isEnabled()
    controls.filter_mode.setCurrentText('Band Pass')
    assert controls.low_frequency.isEnabled()
    controls.low_wavelength.setValue(.1)
    assert controls.options().low_frequency == pytest.approx(10)


def test_window_filter_cannot_accept_band_mode() -> None:
    with pytest.raises(SurfaceAnalysisError, match='带通'):
        SurfaceOptions(filter_type='Average', filter_mode='Band Pass')


@pytest.mark.parametrize('frequency',[.001,.49,.5,.8])
def test_sinusoid_rejects_cutoff_outside_native_defined_bins(frequency: float) -> None:
    options=SurfaceOptions(remove='None',remove_spikes=False,filter_type='FFT Fixed',
        filter_mode='Low Pass',pixel_size_um=1000,high_frequency=frequency,cutoff_shape='Sinusoid')
    with pytest.raises(SurfaceAnalysisError,match='频格'):
        process_surface(np.random.default_rng(3).normal(size=(32,32)),options)


def test_fft_runs_through_page_worker_with_physical_units(qtbot, tmp_path: Path) -> None:
    with np.load(REFERENCE) as reference:
        data, expected = reference['input_1'], reference['expected_1_2']
    path = tmp_path / 'fft.txt'
    np.savetxt(path, data)
    page = PlaneAnalysisPage()
    qtbot.addWidget(page)
    controls = page.surface_controls
    controls.remove.setCurrentIndex(controls.remove.findData('None'))
    controls.spikes.setChecked(False)
    controls.filter_mode.setCurrentText('Band Pass')
    controls.pixel_size.setValue(10)
    controls.low_wavelength.setValue(.1)
    controls.high_wavelength.setValue(.05)
    page.file_edit.setText(str(path))
    page._run_analysis()
    qtbot.waitUntil(lambda: page._result is not None and page._analysis_thread is None, timeout=8000)
    np.testing.assert_allclose(expected, page._result.processed, atol=2e-11, rtol=0)


def test_fft_missing_calibration_reports_error_without_starting_worker(qtbot) -> None:
    page = PlaneAnalysisPage()
    qtbot.addWidget(page)
    page.surface_controls.filter_mode.setCurrentText('Low Pass')
    page.surface_controls.filter_type.setCurrentText('FFT Fixed')
    page.surface_controls.pixel_size.setValue(0)
    timer = QTimer()
    def dismiss_error() -> None:
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QMessageBox):
                widget.accept()
                timer.stop()
    timer.timeout.connect(dismiss_error)
    timer.start(10)
    page._run_analysis()
    timer.stop()
    assert page._analysis_thread is None
    assert '横向标定' in page.status_label.text()
