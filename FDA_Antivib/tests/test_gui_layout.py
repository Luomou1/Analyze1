from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QWidget,
)

import app.gui.main_window as main_window_module
from app.gui.main_window import MainWindow
from app.gui.theme import build_stylesheet, resolve_theme_mode
from app.gui.widgets import NavButton, StepButton, StatusPill, build_nav_icon
from app.plotting.preview import build_normalized_surface_grid


class _FakeSurfaceCanvas(QWidget):
    """测试环境避免初始化 PyVista/OpenGL，仅保留主窗口依赖接口。"""

    def __init__(self) -> None:
        super().__init__()
        self.shutdown_called = False

    def set_info_callback(self, callback) -> None:
        self._info_callback = callback

    def set_pixel_callback(self, callback) -> None:
        self._pixel_callback = callback

    def copy_current_view_to_clipboard(self) -> None:
        return None

    def reset_view(self) -> None:
        return None

    def draw_surface(self, data, title: str) -> None:
        return None

    def draw_comparison(self, left, right, title: str) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture
def safe_main_window(monkeypatch):
    monkeypatch.setattr(main_window_module, "SurfaceCanvas", _FakeSurfaceCanvas)
    monkeypatch.setattr(main_window_module, "ComparisonCanvas", _FakeSurfaceCanvas)
    return MainWindow


def test_heatmap_click_refreshes_pixel_window_across_invalid_pixel(qtbot, safe_main_window, tmp_path):
    from types import SimpleNamespace
    from app.pipeline.session import AnalysisParams
    from app.core.result_model import AnalysisResult
    window = safe_main_window()
    qtbot.addWidget(window)
    z = np.arange(169)*.05
    signal = 1900+1000*np.exp(-.5*((z-4)/.3)**2)*np.cos(22*(z-4))
    cube = np.repeat(np.stack([signal,np.full_like(signal,1900),np.roll(signal,8)])[None],2,axis=0)
    window._analysis_cube = cube
    window._analysis_params = AnalysisParams(tmp_path,0,.05,11,3,'weighted','exe')
    window._result = AnalysisResult(np.zeros((2,3)),np.zeros((2,3)),np.zeros((2,3)),k0_value=11)
    window.h_prime_canvas.draw_map(np.zeros((2,3)), 'height')
    for x in [0,1,2]:
        window.h_prime_canvas._on_click(SimpleNamespace(inaxes=window.h_prime_canvas.axes,xdata=x,ydata=0))
        window._pixel_window.canvas.draw()
        assert f'x={x}, y=0' in window._pixel_window.header_label.text()
        np.testing.assert_allclose(cube[0,x],window._pixel_window.raw_axes.lines[0].get_ydata())


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_build_stylesheet_contains_console_selectors(mode: str) -> None:
    stylesheet = build_stylesheet(mode)

    assert "QFrame#TopCommandBar" in stylesheet
    assert "QFrame#StepRail" in stylesheet
    assert "QFrame#ParameterPanel" in stylesheet
    assert "QPushButton#PrimaryButton" in stylesheet
    assert "QPushButton#TopNavButton" in stylesheet
    assert "QFrame#BottomStatusBar" in stylesheet
    assert "QDialog" in stylesheet


def test_system_theme_falls_back_to_light_without_dark_scheme() -> None:
    assert resolve_theme_mode("system", color_scheme=None) == "light"
    assert resolve_theme_mode("light", color_scheme=None) == "light"
    assert resolve_theme_mode("dark", color_scheme=None) == "dark"


def test_surface_grid_uses_a_fixed_unit_box_without_changing_values() -> None:
    data = np.array([[10.0, 20.0], [30.0, 50.0]], dtype=np.float64)

    grid, display_values, x_coords, y_coords, z_bounds = build_normalized_surface_grid(data)

    assert tuple(grid.bounds) == pytest.approx((0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    assert display_values == pytest.approx(data)
    assert tuple(x_coords) == pytest.approx((0.0, 1.0))
    assert tuple(y_coords) == pytest.approx((0.0, 1.0))
    assert z_bounds == pytest.approx((10.0, 50.0))


def test_navigation_widgets_expose_visual_state(qtbot) -> None:
    nav_button = NavButton(QIcon(), "工作台")
    step_button = StepButton("01", "数据来源")
    status_pill = StatusPill("空闲")
    qtbot.addWidget(nav_button)
    qtbot.addWidget(step_button)
    qtbot.addWidget(status_pill)

    step_button.set_step_state("complete")
    status_pill.set_status("running", "正在分析")

    assert nav_button.property("navRole") == "global"
    assert step_button.property("stepState") == "complete"
    assert status_pill.property("status") == "running"
    assert status_pill.text() == "正在分析"


def test_scan_step_accepts_five_decimal_places(qtbot, safe_main_window) -> None:
    window = safe_main_window()
    qtbot.addWidget(window)
    assert window.step_size.decimals() == 5
    assert window.step_size.minimum() == pytest.approx(0.00001)
    window.step_size.setValue(0.00001)
    assert window.step_size.value() == pytest.approx(0.00001)


def test_surface_tab_builds_only_once_for_latest_result(qtbot, safe_main_window):
    from types import SimpleNamespace
    window = safe_main_window()
    qtbot.addWidget(window)
    calls = []
    window.surface_canvas.draw_surface = lambda data, title: calls.append(data)
    latest = np.arange(12.).reshape(3, 4)
    window._result = SimpleNamespace(h=latest)
    window._pending_surface_tabs = {"surface", "h_prime_surface", "comparison"}
    index = window.tabs.indexOf(window._plot_tabs["surface"])
    window._render_pending_surface_tab(-1)
    assert not calls
    window._render_pending_surface_tab(index)
    window._render_pending_surface_tab(index)
    assert len(calls) == 1
    assert calls[0] is latest
    assert window._pending_surface_tabs == {"h_prime_surface", "comparison"}


@pytest.mark.parametrize("kind", ["workbench", "logs", "settings"])
def test_navigation_icons_are_bundled_and_platform_independent(kind: str, qapp) -> None:
    assert not build_nav_icon(kind).isNull()


def test_main_window_uses_guided_console_layout(qtbot, safe_main_window) -> None:
    window = safe_main_window()
    qtbot.addWidget(window)

    assert window.minimumWidth() == 1180
    assert window.minimumHeight() == 760
    assert window.main_pages.count() == 4
    assert window.parameter_stack.count() == 2
    assert len(window.global_nav_buttons) == 4
    assert [button.text() for button in window.global_nav_buttons] == [
        "工作台",
        "平面分析",
        "台阶分析",
        "设置",
    ]
    assert window.main_pages.widget(1) is window.plane_analysis_page
    assert window.main_pages.widget(2) is window.step_analysis_page
    assert len(window.step_buttons) == 2
    assert window.theme_combo.count() == 3
    assert window.theme_combo.currentData() == "light"
    assert window.image_intensity_mode.currentData() == "mono12_uint16"
    assert window.analysis_window_name.currentData() == "none"
    assert [window.analysis_method.itemData(i) for i in range(window.analysis_method.count())] == [
        "normal", "high", "high1g", "high2g", "gfda",
    ]
    window.analysis_method.setCurrentIndex(window.analysis_method.findData("high2g"))
    assert window.analysis_mode_label.text() == "High 2G"
    assert window.fitting.currentData() == "weighted"
    assert window.unwrap.currentData() == "exe"
    assert window.unwrap.isEnabled()
    assert window.unwrap.findData("unwrap1") >= 0
    assert window.window_size.isEnabled()
    assert not window.fitting.isEnabled()
    window.analysis_method.setCurrentIndex(0)
    assert not hasattr(window, "global_rail")
    assert not hasattr(window, "top_auto_k0_button")
    assert not hasattr(window, "top_start_button")
    assert window.result_content_stack.currentIndex() == 0
    assert not hasattr(window, "stage_label")
    assert not hasattr(window, "command_panel")
    assert not window.auto_k0_button.isEnabled()
    assert not window.start_button.isEnabled()
    assert window.findChildren(QSplitter) == []
    assert window.rail_log.maximumBlockCount() == 120
    assert window.active_range_label.parentWidget() is window.step_rail
    assert window.auto_k0_value_label.parentWidget() is window.step_rail
    assert window.layer_label.parentWidget() is window.step_rail
    assert window.export_text_button.parentWidget() is window.step_rail
    assert window.export_figures_button.parentWidget() is window.step_rail
    assert window.step_rail.layout().itemAt(window.step_rail.layout().count() - 1).widget() is window.rail_log
    assert window.rail_log.frameShape() == QFrame.NoFrame
    assert all(button.isFlat() for button in window.step_buttons)


def test_main_window_closes_all_pyvista_canvases(qtbot, safe_main_window) -> None:
    window = safe_main_window()
    qtbot.addWidget(window)

    window.close()

    assert window.surface_canvas.shutdown_called
    assert window.h_prime_surface_canvas.shutdown_called
    assert window.comparison_canvas.shutdown_called


def test_aux_window_presentation_does_not_steal_focus(qtbot, safe_main_window) -> None:
    class RecordingWindow(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.raise_calls = 0
            self.activate_calls = 0

        def raise_(self) -> None:
            self.raise_calls += 1

        def activateWindow(self) -> None:
            self.activate_calls += 1

    window = safe_main_window()
    aux_window = RecordingWindow()
    qtbot.addWidget(window)
    qtbot.addWidget(aux_window)

    window._present_aux_window(aux_window)

    assert aux_window.isVisible()
    assert aux_window.raise_calls == 1
    assert aux_window.activate_calls == 0

    window._present_aux_window(aux_window)

    assert aux_window.raise_calls == 1
    assert aux_window.activate_calls == 0


def test_step_navigation_preserves_parameter_values(qtbot, safe_main_window) -> None:
    window = safe_main_window()
    qtbot.addWidget(window)
    window.start_height.setValue(12.5)

    window._select_step(1)
    window._select_step(0)

    assert window.parameter_stack.currentIndex() == 0
    assert window.start_height.value() == pytest.approx(12.5)
    assert window.step_buttons[0].isChecked()


def test_data_and_sampling_share_first_step(qtbot, safe_main_window) -> None:
    window = safe_main_window()
    qtbot.addWidget(window)

    window._select_step(0)
    window.step_size.setValue(0.125)

    assert window.parameter_stack.currentIndex() == 0
    assert window.start_height.parent() is not None
    assert window.step_size.value() == pytest.approx(0.125)


def test_run_controls_allow_analysis_without_carrier_preview(qtbot, safe_main_window) -> None:
    window = safe_main_window()
    qtbot.addWidget(window)

    assert not window.auto_k0_button.isEnabled()
    assert not window.start_button.isEnabled()

    window.folder_edit.setText("C:/data/images")
    assert not window.auto_k0_button.isEnabled()
    assert not window.start_button.isEnabled()

    window._select_step(1)
    assert window.auto_k0_button.isEnabled()
    assert window.start_button.isEnabled()

    window.fixed_k0.setValue(0.0)
    assert window.auto_k0_button.isEnabled()
    assert window.start_button.isEnabled()

    window._select_step(1)
    assert window.parameter_stack.currentIndex() == 1
    assert window.auto_k0_button.parent() is not None
    assert window.start_button.parent() is not None
    assert window.parameter_stack.currentWidget().findChildren(QGroupBox) == []
    assert window.parameter_stack.currentWidget().findChildren(QScrollArea) == []


def test_parameter_controls_ignore_wheel_events(qtbot, safe_main_window) -> None:
    window = safe_main_window()
    qtbot.addWidget(window)

    guarded = [
        *window.findChildren(QComboBox),
        *window.findChildren(QAbstractSpinBox),
        *window.findChildren(QPlainTextEdit),
    ]

    assert guarded
    assert all(widget.property("wheelGuard") is True for widget in guarded)
    assert window.eventFilter(window.fixed_k0, QEvent(QEvent.Type.Wheel))


def test_left_rail_log_receives_runtime_messages(qtbot, safe_main_window) -> None:
    window = safe_main_window()
    qtbot.addWidget(window)

    window._append_log("定向验证消息")

    assert "定向验证消息" in window.rail_log.toPlainText()
    assert window.latest_log_label.text() == "定向验证消息"


def test_running_state_locks_parameters_but_keeps_global_navigation(
    qtbot,
    safe_main_window,
) -> None:
    window = safe_main_window()
    qtbot.addWidget(window)

    window._set_running_state(True)

    assert not window.parameter_stack.isEnabled()
    assert not window.step_rail.isEnabled()
    assert window.main_pages.isEnabled()
    assert not window.auto_k0_button.isEnabled()
    assert not window.start_button.isEnabled()
    assert window.auto_k0_button.text() != "任务运行中"
    assert window.start_button.text() == "开始分析"

    window._set_running_state(False)
    assert window.parameter_stack.isEnabled()
    assert window.step_rail.isEnabled()


def test_top_nav_switches_primary_pages(qtbot, safe_main_window) -> None:
    window = safe_main_window()
    qtbot.addWidget(window)

    window.global_nav_buttons[1].click()

    assert window.main_pages.currentIndex() == 1
    assert window.global_nav_buttons[1].isChecked()
    assert window.main_pages.currentWidget() is window.plane_analysis_page

    window.global_nav_buttons[2].click()
    assert window.main_pages.currentWidget() is window.step_analysis_page

    window.global_nav_buttons[3].click()
    assert window.main_pages.currentIndex() == 3
    assert window.global_nav_buttons[3].isChecked()


def test_theme_can_switch_without_a_duplicate_results_page(qtbot, safe_main_window) -> None:
    window = safe_main_window()
    qtbot.addWidget(window)

    window._select_main_page(3)
    dark_index = window.theme_combo.findData("dark")
    window.theme_combo.setCurrentIndex(dark_index)

    assert window.main_pages.currentIndex() == 3
    assert window.log is window.rail_log
    assert window.rail_log.parentWidget() is window.step_rail
    assert "#111827" in QApplication.instance().styleSheet()

    window._select_main_page(0)
    assert window.result_workspace.parentWidget() is window.workbench_result_host


def test_unwrap_and_radius_survive_mode_change(qtbot,safe_main_window):
    window=safe_main_window();qtbot.addWidget(window)
    assert window.window_size.value()==3
    window.unwrap.setCurrentIndex(window.unwrap.findData('unwrap1'))
    window.window_size.setValue(5)
    window.analysis_method.setCurrentIndex(window.analysis_method.findData('high2g'))
    assert window.unwrap.currentData()=='unwrap1'
    assert window.window_size.value()==5


def test_common_k_manual_is_not_overwritten_by_automatic(qtbot,safe_main_window):
    window=safe_main_window();qtbot.addWidget(window)
    assert window.fixed_k0.isEnabled()
    window._show_automatic_common_k0(11.2)
    assert not window._manual_common_k0
    window.fixed_k0.setValue(10.7)
    assert window._manual_common_k0
    window._show_automatic_common_k0(11.3)
    assert window.fixed_k0.value()==10.7
    window.fixed_k0.setValue(0)
    assert not window._manual_common_k0
    window._show_automatic_common_k0(11.3)
    assert window.fixed_k0.value()==11.3


def test_explicit_auto_k0_completion_replaces_manual_value(qtbot,safe_main_window):
    window=safe_main_window();qtbot.addWidget(window)
    window.fixed_k0.setValue(10.7)
    assert window._manual_common_k0
    result={'k0_value':11.123456,'k_axis':np.array([10.,11.,12.]),
            'spectrum':np.array([1.,3.,1.]),'peak_index':1,'peak_prominence':3.,
            'window_name':'none','zero_padding_mode':'none','candidate_count':100}
    window._handle_auto_k0_finished(result)
    assert window.fixed_k0.value()==11.123456
    assert not window._manual_common_k0
    window.fixed_k0.setValue(10.9)
    assert window._manual_common_k0


def test_high2g_diagnostics_are_created_only_on_request(qtbot,safe_main_window):
    from app.core.result_model import AnalysisResult
    window=safe_main_window();qtbot.addWidget(window)
    window.analysis_method.setCurrentIndex(window.analysis_method.findData('high2g'))
    data=np.arange(20,dtype=float).reshape(4,5)
    result=AnalysisResult(data,data+.1,data,k0_value=11,
                          extras={'analysis_method':'high2g','theta_map':data})
    window._handle_finished(result)
    assert 'theta' not in window._plot_tabs
    assert not hasattr(window,'theta_canvas')
    assert window.high2g_diagnostics_button.isEnabled()
    window.high2g_diagnostics_button.click()
    assert window.tabs.indexOf(window._plot_tabs['theta'])>=0
    np.testing.assert_allclose(data/(2*np.pi),window.theta_canvas._data)
    window.high2g_diagnostics_button.click()
    assert window.tabs.indexOf(window._plot_tabs['theta'])==-1
    window.high2g_diagnostics_button.click()
    window._handle_finished(result)
    assert not window._high2g_diagnostics_visible
    assert window.tabs.indexOf(window._plot_tabs['theta'])==-1
