"""GFDA 的标定文件和扫描运动诊断，保留最近一次正式运行快照。"""
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from app.reconstruction.gfda_calibration import Calibration, save_calibration
from app.reconstruction.gfda_phase import GfdaError
from app.gui.mpl_font import configure_matplotlib_fonts


class GfdaControls(QWidget):
    """空路径表示本次扫描自动标定；保存操作只保存已计算成功的校正表。"""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._calibration: Calibration | None = None
        self._scan_window: QWidget | None = None
        self._result: Mapping | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.hint = QLabel("名义步长用于建立长度尺度；GFDA 从图像估计实际扫描位置。需要足够的空间条纹。")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        layout.addWidget(QLabel("GFDA 校正文件（留空时按本次数据生成）"))
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("本次扫描自动标定")
        self.path_edit.setAccessibleName("GFDA校正文件")
        self.path_edit.textChanged.connect(lambda _text: self.changed.emit())
        row.addWidget(self.path_edit, 1)
        choose = QPushButton("选择")
        choose.clicked.connect(self._choose)
        row.addWidget(choose)
        clear = QPushButton("清空")
        clear.clicked.connect(self.path_edit.clear)
        row.addWidget(clear)
        layout.addLayout(row)
        actions = QHBoxLayout()
        self.save_button = QPushButton("保存本次校正")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)
        actions.addWidget(self.save_button)
        self.scan_button = QPushButton("扫描坐标诊断")
        self.scan_button.setEnabled(False)
        self.scan_button.clicked.connect(self._show_scan)
        actions.addWidget(self.scan_button)
        layout.addLayout(actions)
        self.status = QLabel("运行 GFDA 后可保存校正表并查看扫描坐标。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    @property
    def calibration_path(self) -> Path | None:
        value = self.path_edit.text().strip()
        return Path(value) if value else None

    def _choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 GFDA 校正文件", "",
                    "GFDA 校正 (*.json *.avc);;所有文件 (*)")
        if path:
            self.path_edit.setText(path)

    def _save(self) -> None:
        if self._calibration is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存本次 GFDA 校正", "scan.gfda.json",
                                             "GFDA 校正 (*.json)")
        if path:
            try:
                save_calibration(path, self._calibration)
            except (OSError, GfdaError) as error:
                QMessageBox.critical(self, "校正文件保存失败", str(error))
                return
            self.status.setText(f"已保存本次校正：{path}")

    def set_result(self, result: Mapping) -> None:
        """正式结果切换时同步替换校正与诊断，不从当前控件重新推导坐标。"""
        self._result = result if result.get('gfda_applied') else None
        self._calibration = result.get('gfda_calibration') if self._result is not None else None
        self.save_button.setEnabled(self._calibration is not None)
        self.scan_button.setEnabled(self._result is not None)
        if self._scan_window is not None:
            self._scan_window.close()
            self._scan_window.deleteLater()
            self._scan_window = None
        if self._result is not None:
            confidence = np.asarray(result['scan_step_confidence'])
            valid = np.asarray(result['valid_mask'])
            self.status.setText(f"已执行 GFDA：观测 {np.count_nonzero(confidence)}/{len(confidence)} 个步进，"
                                f"有效像素 {valid.mean():.1%}；零置信度区为包络外名义延拓。")
        else:
            self.status.setText("运行 GFDA 后可保存校正表并查看扫描坐标。")

    def _show_scan(self) -> None:
        if self._result is None:
            return
        if self._scan_window is None:
            self._scan_window = ScanWindow(self._result, self.window())
            self._scan_window.setWindowTitle("GFDA 扫描坐标诊断")
        self._scan_window.show()
        self._scan_window.raise_()

    def close_diagnostics(self) -> None:
        if self._scan_window is not None:
            self._scan_window.close()
            self._scan_window.deleteLater()
            self._scan_window = None


class ScanWindow(QWidget):
    """显示原始估计、一次平滑结果和每个步进的观测置信度。"""

    def __init__(self, result: Mapping, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Tool)
        self.resize(880, 650)
        layout = QVBoxLayout(self)
        figure = Figure(figsize=(8, 6), constrained_layout=True)
        font = configure_matplotlib_fonts()
        canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(canvas)
        top, middle, bottom = figure.subplots(3, 1, sharex=True)
        nominal = np.asarray(result['scan_positions_raw_um'])
        raw = np.asarray(result['scan_positions_estimated_raw_um'])
        used = np.asarray(result['scan_positions_used_um'])
        frames = np.arange(len(nominal))+1
        top.plot(frames, (raw-nominal)*1000, label='平滑前估计')
        top.plot(frames, (used-nominal)*1000, label='实际使用')
        top.set_ylabel('位置修正 (nm)', fontproperties=font)
        top.legend(prop=font)
        middle.plot(frames[1:], np.diff(nominal)*1000, label='名义步进')
        middle.plot(frames[1:], np.diff(used)*1000, label='实际使用')
        middle.set_ylabel('步进 (nm)', fontproperties=font)
        middle.legend(prop=font)
        confidence = np.asarray(result['scan_step_confidence'])
        bottom.plot(frames[1:], confidence)
        bottom.set_ylim(-.05, 1.05)
        bottom.set_ylabel('观测置信度', fontproperties=font)
        bottom.set_xlabel('帧号', fontproperties=font)
        for axes in (top, middle, bottom):
            axes.grid(alpha=.2)
        canvas.draw_idle()
