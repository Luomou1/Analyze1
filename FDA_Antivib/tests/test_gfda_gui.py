"""GFDA 工作流选择、标定和正式诊断的 GUI 集成。"""
from types import SimpleNamespace

import numpy as np

from app.core.result_model import AnalysisResult
from app.gui.gfda_controls import GfdaControls
from app.pipeline.session import AnalysisParams
from app.reconstruction.engine import analyze
from app.reconstruction.gfda_calibration import load_calibration
from test_gfda import moving_surface
from test_gui_layout import safe_main_window


def test_gfda_is_peer_method_and_nominal_step_stays_enabled(qtbot, safe_main_window, tmp_path):
    window = safe_main_window()
    qtbot.addWidget(window)
    window.folder_edit.setText(str(tmp_path))
    window.analysis_method.setCurrentIndex(window.analysis_method.findData('gfda'))
    assert window.analysis_mode_label.text() == 'GFDA'
    assert window.step_size.isEnabled()
    assert '名义' in window.step_size_label.text()
    assert not window.unwrap.isEnabled()
    assert window._is_high2g_mode()
    window.gfda_controls.path_edit.setText(str(tmp_path/'scan.gfda.json'))
    params = window._build_params()
    assert params.analysis_method == 'gfda'
    assert params.gfda_calibration_path == tmp_path/'scan.gfda.json'
    window.analysis_method.setCurrentIndex(window.analysis_method.findData('normal'))
    assert window.unwrap.isEnabled()
    assert window.gfda_controls.isHidden()


def test_gfda_gui_result_pixel_scan_and_calibration_save(qtbot, safe_main_window, monkeypatch, tmp_path):
    import app.gui.gfda_controls as controls_module
    cube, _, _ = moving_surface()
    result = analyze(cube, .05, mode='gfda')
    window = safe_main_window()
    qtbot.addWidget(window)
    window.analysis_method.setCurrentIndex(window.analysis_method.findData('gfda'))
    window._analysis_params = AnalysisParams(tmp_path,0,.05,None,3,'weighted','exe',analysis_method='gfda')
    window._worker = SimpleNamespace(last_cube=cube)
    window._handle_finished(AnalysisResult.from_mapping(result))
    assert window.gfda_controls.save_button.isEnabled()
    assert window.gfda_controls.scan_button.isEnabled()
    window._handle_pixel_click('h_prime', {'x':20.,'y':5.})
    np.testing.assert_array_equal(result['scan_positions_used_um'], window._pixel_window.raw_axes.lines[0].get_xdata())
    window.gfda_controls.scan_button.click()
    assert window.gfda_controls._scan_window.isVisible()
    path = tmp_path/'saved.gfda.json'
    monkeypatch.setattr(controls_module.QFileDialog, 'getSaveFileName', lambda *a, **k: (str(path), ''))
    window.gfda_controls.save_button.click()
    calibration = result['gfda_calibration']
    loaded = load_calibration(path, .05, calibration.carrier, calibration.phase_half_width)
    np.testing.assert_array_equal(calibration.coefficients, loaded.coefficients)
    window.close()
    assert window.gfda_controls._scan_window is None


def test_calibration_path_change_invalidates_k0_snapshot(qtbot, safe_main_window):
    window = safe_main_window()
    qtbot.addWidget(window)
    window._auto_k0_result = {'anything': 'old'}
    window.gfda_controls.path_edit.setText('new.gfda.json')
    assert window._auto_k0_result is None


def test_gfda_runs_image_files_through_real_worker_thread(qtbot, safe_main_window, monkeypatch, tmp_path):
    from PIL import Image
    import app.gui.main_window as window_module
    cube, _, _ = moving_surface()
    integer = np.rint(cube).astype(np.uint16)
    for frame in range(integer.shape[-1]):
        # 项目 Mono12 文件约定为左对齐 uint16，读取时右移 4 位。
        Image.fromarray(integer[..., frame] << 4).save(tmp_path/f'frame_{frame:04d}.png')
    window = safe_main_window()
    qtbot.addWidget(window)
    errors = []
    monkeypatch.setattr(window_module.QMessageBox, 'critical', lambda *args: errors.append(args[-1]))
    window.folder_edit.setText(str(tmp_path))
    window.analysis_method.setCurrentIndex(window.analysis_method.findData('gfda'))
    window._start_analysis()
    qtbot.waitUntil(lambda: window._thread is None, timeout=30000)
    assert not errors
    assert window._result.extras['gfda_applied']
    assert window._analysis_cube.shape == cube.shape
    expected = analyze(integer, .05, mode='gfda')
    np.testing.assert_allclose(expected['h_prime'], window._result.h_prime, atol=.001)


def test_closing_while_worker_runs_keeps_window_alive(qtbot, safe_main_window):
    window = safe_main_window()
    qtbot.addWidget(window)
    window._thread = SimpleNamespace(isRunning=lambda: True)
    assert not window.close()
    assert '仍在运行' in window.status_label.text()
    window._thread = None
