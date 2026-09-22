from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.gui.surface_analysis_page import (
    PlaneAnalysisPage,
    ResultFigureCanvas,
    Surface3DCanvas,
    StepAnalysisPage,
    StepSelectionCanvas,
    _downsample_surface,
    _height_limits,
)


def test_fill_method_switch_preserves_valid_user_options(qtbot):
    from app.gui.surface_controls import SurfaceControls
    controls = SurfaceControls()
    qtbot.addWidget(controls)
    controls.fill.setChecked(True)
    assert 'MetroPro EXE' == controls.options().data_fill_method
    assert not controls.fill_mode.isEnabled()
    controls.fill_method.setCurrentIndex(controls.fill_method.findData('Polynomial'))
    controls.fill_mode.setCurrentIndex(2)
    assert 2 == controls.options().data_fill_mode
    controls.fill_method.setCurrentIndex(controls.fill_method.findData('MetroPro EXE'))
    assert 0 == controls.options().data_fill_mode
    assert not controls.fill_mode.isEnabled()


def test_height_limits_center_bulk_without_clipping_extrema() -> None:
    data = np.zeros((20, 20))
    data[1, 1] = 100
    data[2, 2] = np.nan
    lower, upper = _height_limits(data)
    assert lower == pytest.approx(-110)
    assert upper == pytest.approx(110)
    assert _height_limits(np.full((3, 3), 5000.0)) == (4999.5, 5000.5)


def test_plane_export_round_trips_processed_matrix(qtbot, tmp_path, monkeypatch) -> None:
    from app.gui.surface_analysis_page import QFileDialog
    data = np.arange(120, dtype=float).reshape(10, 12)
    data[3, 4] = np.nan
    source = tmp_path / 'plane.txt'
    target = tmp_path / 'processed.txt'
    np.savetxt(source, data)
    page = PlaneAnalysisPage()
    qtbot.addWidget(page)
    assert not page.export_button.isEnabled()
    page.file_edit.setText(str(source))
    page.surface_controls.spikes.setChecked(False)
    page._run_analysis()
    qtbot.waitUntil(lambda: page._result is not None and page._analysis_thread is None, timeout=8000)
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a: (str(target), ''))
    page.export_button.click()
    np.testing.assert_equal(page._result.processed, np.loadtxt(target))
    page.processed_surface_canvas._render_enabled = False
    page._render_current_tab(1)
    lower, upper = page.processed_surface_canvas._last_display_bounds[4:]
    assert lower < 0 < upper
    assert upper - lower < 2
    page.conversion_factor.setValue(2)
    assert not page.export_button.isEnabled()


@pytest.mark.parametrize('cancel', [True, False])
def test_plane_export_cancel_and_write_failure_keep_result(qtbot, tmp_path, monkeypatch, cancel) -> None:
    from app.core.surface_analysis import analyze_plane
    from app.core.surface_options import SurfaceOptions
    from app.gui.surface_analysis_page import QFileDialog, QMessageBox
    page = PlaneAnalysisPage()
    qtbot.addWidget(page)
    page.file_edit.setText(str(tmp_path / 'input.txt'))
    result = analyze_plane(np.arange(24.0).reshape(4, 6), SurfaceOptions(remove_spikes=False))
    page._handle_analysis_finished(result)
    page._finalize_analysis_thread()
    errors = []
    path = '' if cancel else str(tmp_path / 'missing' / 'output.txt')
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a: (path, ''))
    monkeypatch.setattr(QMessageBox, 'critical', lambda *a: errors.append(a[2]))
    page.export_button.click()
    assert len(errors) == (0 if cancel else 1)
    assert page._result is result
    assert page.export_button.isEnabled()


def test_plane_and_step_pages_keep_independent_file_inputs(qtbot) -> None:
    plane_page = PlaneAnalysisPage()
    step_page = StepAnalysisPage()
    qtbot.addWidget(plane_page)
    qtbot.addWidget(step_page)

    plane_page.file_edit.setText("C:/data/plane.txt")
    step_page.file_edit.setText("C:/data/step.txt")

    assert plane_page.file_edit.text().endswith("plane.txt")
    assert step_page.file_edit.text().endswith("step.txt")
    assert plane_page.file_edit is not step_page.file_edit


def test_step_page_exposes_independent_spike_control(qtbot) -> None:
    page = StepAnalysisPage()
    qtbot.addWidget(page)

    assert page.surface_controls.spikes.isChecked()
    page.surface_controls.spikes.setChecked(False)
    assert not page.surface_controls.spikes.isChecked()
    assert not page.analyze_button.isEnabled()


def test_step_can_run_without_three_points(qtbot, tmp_path) -> None:
    data = np.zeros((12, 16)); data[:, 8:] = 20
    path = tmp_path / 'step.txt'; np.savetxt(path, data)
    page = StepAnalysisPage(); qtbot.addWidget(page)
    page.surface_controls.spikes.setChecked(False)
    page.file_edit.setText(str(path)); page._load_for_selection()
    assert page.analyze_button.isEnabled()
    assert not page.selection_canvas.points
    page._run_analysis()
    qtbot.waitUntil(lambda: page._result is not None and page._analysis_thread is None, timeout=8000)
    np.testing.assert_equal(data, page._result.processed)


def test_step_report_export_and_invalidation(qtbot, tmp_path, monkeypatch):
    import json
    from app.gui.surface_analysis_page import QFileDialog
    data = np.zeros((12, 16)); data[:, 8:] = 20
    source, target = tmp_path/'step.txt', tmp_path/'step.npz'
    np.savetxt(source, data)
    page = StepAnalysisPage(); qtbot.addWidget(page)
    assert not page.export_button.isEnabled()
    page.surface_controls.spikes.setChecked(False)
    page.file_edit.setText(str(source)); page._load_for_selection(); page._run_analysis()
    qtbot.waitUntil(lambda: page._result is not None and page._analysis_thread is None, timeout=8000)
    page._on_regions_changed(((1, 1, 6, 10), (10, 1, 15, 10)))
    assert '平台一 Sa' in page.metrics.toPlainText()
    page.region_reference.setCurrentIndex(2)
    assert page._measurement.step_height == -20
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a: (str(target), ''))
    page.export_button.click()
    with np.load(target, allow_pickle=False) as saved:
        report = json.loads(str(saved['report_json']))
        assert 'plane' == report['regions']['reference']
        assert -20 == report['regions']['measurement']['step_height']
    page._reset_regions()
    assert page._region_report is None
    page.surface_controls.spikes.setChecked(True)
    assert not page.export_button.isEnabled()


def test_step_failure_does_not_suggest_unused_three_points(qtbot, monkeypatch):
    from app.gui.surface_analysis_page import QMessageBox
    page = StepAnalysisPage(); qtbot.addWidget(page)
    errors = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: errors.append(args[2]))
    page._handle_analysis_failed('没有有效点')
    assert '三个' not in errors[0]


@pytest.mark.parametrize('page_type', [PlaneAnalysisPage, StepAnalysisPage])
def test_running_analysis_defers_page_and_main_window_close(qtbot, page_type):
    from PySide6.QtCore import QSemaphore, QThread
    from PySide6.QtGui import QCloseEvent
    from app.gui.main_window import MainWindow

    gate = QSemaphore()
    class WaitingThread(QThread):
        def run(self):
            gate.acquire()

    page = page_type(); qtbot.addWidget(page)
    thread = WaitingThread()
    page._analysis_thread = thread
    thread.start()
    try:
        qtbot.waitUntil(thread.isRunning)
        event = QCloseEvent()
        page.closeEvent(event)
        assert not event.isAccepted()
        main_event = QCloseEvent()
        MainWindow.closeEvent(SimpleNamespace(plane_analysis_page=page, step_analysis_page=None), main_event)
        assert not main_event.isAccepted()
        assert thread.isRunning()
    finally:
        gate.release()
        assert thread.wait(3000)
        page._analysis_thread = None


def test_step_export_failure_and_cancel_preserve_result(qtbot, tmp_path, monkeypatch):
    from app.core.surface_analysis import analyze_step
    from app.core.surface_options import SurfaceOptions
    from app.gui.surface_analysis_page import QFileDialog, QMessageBox
    page = StepAnalysisPage(); qtbot.addWidget(page)
    result = analyze_step(np.ones((8, 10)), options=SurfaceOptions(remove='None', remove_spikes=False))
    page._handle_analysis_finished(result)
    errors = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: errors.append(args[2]))
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: ('', ''))
    page.export_button.click()
    assert not errors
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(tmp_path/'missing'/'report.npz'), ''))
    page.export_button.click()
    assert len(errors) == 1
    assert page._result is result
    assert page.export_button.isEnabled()


def test_filter_window_controls_follow_filter_mode(qtbot) -> None:
    page = PlaneAnalysisPage(); qtbot.addWidget(page)
    assert not page.surface_controls.window.isEnabled()
    page.surface_controls.filter_mode.setCurrentText('Low Pass')
    assert page.surface_controls.window.isEnabled()
    assert page.surface_controls.filter_trim.isEnabled()


def test_analysis_pages_expose_accessible_primary_controls(qtbot) -> None:
    plane_page = PlaneAnalysisPage()
    step_page = StepAnalysisPage()
    qtbot.addWidget(plane_page)
    qtbot.addWidget(step_page)

    assert plane_page.choose_button.accessibleName() == "选择平面高度文件"
    assert plane_page.analyze_button.accessibleName() == "开始平面分析"
    assert step_page.choose_button.accessibleName() == "选择台阶高度文件"
    assert step_page.analyze_button.accessibleName() == "开始台阶分析"


def test_result_charts_use_one_tab_per_chart(qtbot) -> None:
    plane_page = PlaneAnalysisPage()
    step_page = StepAnalysisPage()
    qtbot.addWidget(plane_page)
    qtbot.addWidget(step_page)

    assert [plane_page.tabs.tabText(index) for index in range(plane_page.tabs.count())] == [
        "原始三维",
        "处理后三维",
        "二维高度图",
        "一维轮廓",
    ]
    assert [step_page.tabs.tabText(index) for index in range(step_page.tabs.count())] == [
        "交互选择",
        "原始三维",
        "分层结果",
        "处理后三维",
        "分层标准差",
        "一维轮廓",
    ]


def test_surface_preview_is_capped_to_a_small_render_grid() -> None:
    data = np.zeros((1024, 1280), dtype=float)

    reduced, x, y = _downsample_surface(data)

    assert reduced.shape[0] <= 160
    assert reduced.shape[1] <= 160
    assert x.size == reduced.shape[1]
    assert y.size == reduced.shape[0]


def test_surface_canvas_can_reuse_original_z_limits(qtbot) -> None:
    canvas = Surface3DCanvas(render_enabled=False)
    qtbot.addWidget(canvas)

    canvas.draw_surface(np.ones((8, 10), dtype=float) * 12.0, "处理后三维", z_limits=(0.0, 100.0))

    assert canvas._last_display_bounds is not None
    assert canvas._last_display_bounds[4:] == (0.0, 100.0)


def test_profile_canvas_reports_two_clicked_points_delta(qtbot) -> None:
    canvas = ResultFigureCanvas()
    qtbot.addWidget(canvas)
    data = np.array([[0.0, 2.0, 5.0, 9.0, 13.0]], dtype=float)

    canvas.draw_profile(data, "row", 0)
    axes = canvas.figure.axes[0]
    canvas._on_profile_click(SimpleNamespace(inaxes=axes, xdata=1.0, ydata=2.0))
    canvas._on_profile_click(SimpleNamespace(inaxes=axes, xdata=4.0, ydata=13.0))

    overlay_text = "\n".join(
        artist.get_text() for artist in canvas._profile_artists if hasattr(artist, "get_text")
    )
    assert "ΔH=11.000 nm" in overlay_text


def test_step_selection_canvas_displays_first_matrix_row_at_top(qtbot) -> None:
    canvas = StepSelectionCanvas()
    qtbot.addWidget(canvas)
    canvas.start_point_selection(np.arange(24, dtype=float).reshape(4, 6))

    assert canvas.axes.images[0].origin == "upper"


def test_profile_spinboxes_are_configured_for_fast_adjustment(qtbot) -> None:
    plane_page = PlaneAnalysisPage()
    step_page = StepAnalysisPage()
    qtbot.addWidget(plane_page)
    qtbot.addWidget(step_page)

    assert plane_page.profile_index.isAccelerated()
    assert step_page.profile_index.isAccelerated()
    assert not plane_page.profile_index.keyboardTracking()
    assert not step_page.profile_index.keyboardTracking()
    assert plane_page.profile_index.buttonSymbols().name == "NoButtons"
    assert step_page.profile_index.buttonSymbols().name == "NoButtons"
    assert plane_page.profile_index.minimumWidth() >= 112
    assert step_page.profile_index.minimumWidth() >= 112


def test_plane_and_step_use_pyvista_for_3d_surfaces(qtbot) -> None:
    plane_page = PlaneAnalysisPage()
    step_page = StepAnalysisPage()
    qtbot.addWidget(plane_page)
    qtbot.addWidget(step_page)

    assert isinstance(plane_page.raw_surface_canvas, Surface3DCanvas)
    assert isinstance(plane_page.processed_surface_canvas, Surface3DCanvas)
    assert isinstance(step_page.raw_surface_canvas, Surface3DCanvas)
    assert isinstance(step_page.processed_surface_canvas, Surface3DCanvas)


@pytest.mark.parametrize("denoise", [True, False])
def test_plane_page_runs_one_independent_height_file(qtbot, tmp_path, denoise) -> None:
    y, x = np.mgrid[:24, :30]
    data = 0.4 * x - 0.2 * y + 50.0 + 0.5 * np.sin(x / 4.0)
    path = tmp_path / "plane.txt"
    np.savetxt(path, data)

    page = PlaneAnalysisPage()
    qtbot.addWidget(page)
    page.file_edit.setText(str(path))
    page.surface_controls.spikes.setChecked(denoise)
    page._run_analysis()
    qtbot.waitUntil(lambda: page._result is not None and page._analysis_thread is None, timeout=8000)

    assert page._result is not None
    assert page._result.original.shape == data.shape
    if not denoise:
        np.testing.assert_array_equal(data, page._result.original)
        assert page._result.noise_count == 0
    assert "高度范围" in page.metrics.toPlainText()
    assert page._rendered_tabs == {2}


@pytest.mark.parametrize("denoise", [True, False])
def test_step_page_runs_and_measures_regions_with_same_three_points(qtbot, tmp_path, denoise) -> None:
    y, x = np.mgrid[:36, :48]
    data = 0.1 * x - 0.08 * y + np.where(x >= 24, 40.0, 0.0)
    path = tmp_path / "step.txt"
    np.savetxt(path, data)

    page = StepAnalysisPage()
    qtbot.addWidget(page)
    page.file_edit.setText(str(path))
    page.surface_controls.remove.setCurrentIndex(page.surface_controls.remove.findData("Three Points"))
    page._load_for_selection()
    page.selection_canvas._points = [(5, 5), (28, 8), (12, 20)]
    page._on_points_changed(page.selection_canvas.points)
    page.surface_controls.spikes.setChecked(denoise)
    page._run_analysis()
    qtbot.waitUntil(lambda: page._result is not None and page._analysis_thread is None, timeout=8000)

    assert page._result is not None
    assert page._result.three_point
    assert page.analyze_button.isEnabled()
    assert page._rendered_tabs == set()

    page._on_regions_changed(((28, 5, 42, 28), (4, 5, 18, 28)))

    assert page._measurement is not None
    assert page._measurement.step_height > 30.0
    assert "台阶平均高度差" in page.metrics.toPlainText()
    page._render_current_tab(2)
    assert len(page.layer_map_canvas.figure.axes[0].collections) > 0
