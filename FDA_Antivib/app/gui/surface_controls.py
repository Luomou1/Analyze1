"""平面/台阶共用的表面处理控制面板。"""
from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QSpinBox, QVBoxLayout, QWidget

from app.core.surface_options import SurfaceOptions
from app.gui.surface_zernike_controls import ZernikeControls


class SurfaceControls(QWidget):
    changed = Signal()

    def __init__(self, *, step: bool = False) -> None:
        super().__init__()
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        self.remove = QComboBox()
        for label, value in [('不去除', 'None'), ('高度偏置 Piston', 'Piston'), ('平面 Plane', 'Plane'),
                             ('球面 Sphere', 'Sphere'), ('柱面 Cylinder', 'Cylinder'),
                             ('四阶多项式 4th Order', '4th Order')]:
            self.remove.addItem(label, value)
        if step:
            self.remove.addItem('三点调平（同层参考点）', 'Three Points')
        for name in ('Fixed Rad Sphere', 'Fixed Angle Cone', 'Variable Angle Cone'):
            self.remove.addItem(name, name)
        self.remove.addItem('泽尼克 Zernike Fringe', 'Zernike Fringe')
        self.remove.setCurrentIndex(0 if step else 2)
        form = self._group(layout, '形状与有效区域')
        form.addRow('去除形状 Remove', self.remove)
        self.radius = QDoubleSpinBox(); self.radius.setSuffix(' mm'); self.radius.setDecimals(6)
        self.radius.setRange(-1e6,1e6)
        self.radius.setToolTip('输入已知球面半径；正值表示凸面，负值表示凹面。')
        form.addRow('球面半径', self.radius)
        self.cone_angle = QDoubleSpinBox(); self.cone_angle.setRange(-180,180)
        self.cone_angle.setDecimals(6); self.cone_angle.setSuffix(' °')
        self.cone_angle.setToolTip('全锥角：凸面为正、凹面为负；范围 ±5～±180°，自由角度填 0 自动估计。')
        form.addRow('全锥角 Cone Angle',self.cone_angle)
        self.zernike = ZernikeControls()
        self.zernike.changed.connect(self._changed)
        form.addRow(self.zernike)
        self.trim = self._integer(0, 10, 0)
        self.trim_mode = QComboBox(); self.trim_mode.addItems(['All', 'Outside'])
        form.addRow('边缘裁剪 Trim（层）', self.trim)
        form.addRow('裁剪范围', self.trim_mode)
        self.min_area = self._integer(0, 9999999, 0)
        self.min_area.setSpecialValueText('关闭')
        self.min_area.setToolTip('保留面积不小于阈值的四连通有效点区域；在 Trim、去尖峰之后、滤波之前执行。')
        form.addRow('最小区域 Min Area Size（像素）', self.min_area)
        self.spikes = QCheckBox('去除尖峰 Remove Spikes'); self.spikes.setChecked(True)
        self.spike_height = QDoubleSpinBox(); self.spike_height.setRange(.01, 100); self.spike_height.setValue(2.5)
        form.addRow(self.spikes); form.addRow('尖峰阈值 ×RMS', self.spike_height)
        self.fill = QCheckBox('缺失点补偿 Data Fill')
        self.fill_method = QComboBox()
        self.fill_method.addItem('原厂应用 MetroPro EXE', 'MetroPro EXE')
        self.fill_method.addItem('局部二次面 Polynomial', 'Polynomial')
        self.fill_method.addItem('微分松弛 Differential', 'Differential')
        self.fill_mode = QComboBox()
        self.fill_mode.addItem('限面积内部孔', 0)
        self.fill_mode.addItem('全部内部孔（忽略面积上限）', 1)
        self.fill_mode.addItem('含边界孔（忽略面积上限）', 2)
        self.fill_mode.setToolTip('边界补洞可能外推高度；测量时应检查原始有效域。')
        self.fill_max = self._integer(0, 1000000, 25)
        form.addRow(self.fill); form.addRow('补洞算法', self.fill_method)
        form.addRow('补洞范围', self.fill_mode)
        form.addRow('最大孔洞（像素数）', self.fill_max)
        form = self._group(layout, '空间滤波')
        self.filter_mode = QComboBox(); self.filter_mode.addItems(['Off', 'Low Pass', 'High Pass', 'Band Pass', 'Band Reject'])
        self.filter_type = QComboBox(); self.filter_type.addItems(['Average', 'Median', '2 Sigma', 'FFT Fixed', 'Gauss Spline', 'Robust Gauss Spline'])
        self.filter_type.addItems(['FFT Auto', 'Gauss Spline Auto', 'Robust Gauss Spline Auto'])
        self.window = self._integer(3, 99, 3); self.window.setSingleStep(2)
        self.filter_trim = QCheckBox('裁去滤波边缘 Filter Trim')
        form.addRow('滤波 Filter', self.filter_mode); form.addRow('算法 Filter Type', self.filter_type)
        form.addRow('窗口边长（像素）', self.window); form.addRow(self.filter_trim)
        self.pixel_size = self._decimal(' μm/pixel')
        self.pixel_size.setValue(SurfaceOptions().pixel_size_um)
        self.pixel_size.setSpecialValueText('未标定')
        self.pixel_size.setToolTip('填写样品面上的横向像素尺寸；不是扫描步长。')
        form.addRow('横向标定', self.pixel_size)
        self.cutoff_shape = QComboBox(); self.cutoff_shape.addItem('Gaussian')
        self.cutoff_shape.addItem('Sinusoid')
        form.addRow('截止形状 Filter Cutoff', self.cutoff_shape)
        self.low_frequency = self._decimal(' 1/mm')
        self.high_frequency = self._decimal(' 1/mm')
        self.low_wavelength = self._decimal(' mm')
        self.high_wavelength = self._decimal(' mm')
        for name, field in (('Low Wavelen', self.low_wavelength), ('High Wavelen', self.high_wavelength),
                            ('Low Freq', self.low_frequency), ('High Freq', self.high_frequency)):
            field.setToolTip('Low/High 指频率截止端；同名波长为频率倒数。0 表示未设置。')
            form.addRow(name, field)
        self.low_frequency.valueChanged.connect(lambda value: self._reciprocal(value, self.low_wavelength))
        self.high_frequency.valueChanged.connect(lambda value: self._reciprocal(value, self.high_wavelength))
        self.low_wavelength.valueChanged.connect(lambda value: self._reciprocal(value, self.low_frequency))
        self.high_wavelength.valueChanged.connect(lambda value: self._reciprocal(value, self.high_frequency))
        notice = QLabel('Auto 提供起始截止值，实际值随数据尺寸确定；请按测量目的检查结果。')
        notice.setWordWrap(True); form.addRow(notice)
        self.filter_trim.setToolTip('开启：仅保留窗口内全部点有效的位置；关闭：按有效邻点计算并保留原有效域。')
        for widget in (self.remove, self.trim_mode, self.filter_mode, self.filter_type, self.cutoff_shape, self.fill_mode, self.fill_method):
            widget.currentIndexChanged.connect(self._changed)
        for widget in (self.trim, self.spike_height, self.fill_max, self.window, self.pixel_size, self.min_area, self.radius, self.cone_angle):
            widget.valueChanged.connect(self._changed)
        for widget in (self.spikes, self.fill, self.filter_trim):
            widget.toggled.connect(self._changed)
        self._changed()

    @staticmethod
    def _decimal(suffix: str) -> QDoubleSpinBox:
        field = QDoubleSpinBox()
        field.setDecimals(9); field.setRange(0, 1e9); field.setSuffix(suffix)
        field.setKeyboardTracking(False)
        return field

    def _reciprocal(self, value: float, target: QDoubleSpinBox) -> None:
        with QSignalBlocker(target):
            target.setValue(1/value if value > 0 else 0)
        self.changed.emit()

    def show_resolved_options(self, options: SurfaceOptions | None) -> None:
        """显示本次 Auto 实际截止值，避免发出参数修改信号使刚完成的结果失效。"""
        if options is None or options.filter_mode == 'Off' or not options.filter_type.endswith(' Auto'):
            return
        for field, value in ((self.low_frequency,options.low_frequency),
                             (self.high_frequency,options.high_frequency),
                             (self.low_wavelength,1/options.low_frequency),
                             (self.high_wavelength,1/options.high_frequency)):
            with QSignalBlocker(field):
                field.setValue(value)

    @staticmethod
    def _pending(combo: QComboBox, name: str) -> None:
        combo.addItem(name + '（待验证）', name)
        item = combo.model().item(combo.count()-1)
        item.setEnabled(False)
        item.setToolTip('尚未完成原厂算法对照，暂不可用。')

    @staticmethod
    def _integer(low: int, high: int, value: int) -> QSpinBox:
        widget = QSpinBox(); widget.setRange(low, high); widget.setValue(value)
        return widget

    @staticmethod
    def _group(layout: QVBoxLayout, title: str) -> QFormLayout:
        group = QGroupBox(title)
        group.setObjectName('SurfaceControlGroup')
        form = QFormLayout(group); layout.addWidget(group)
        return form

    def _changed(self) -> None:
        self.zernike.setVisible(self.remove.currentData() == 'Zernike Fringe')
        exe_fill = self.fill_method.currentData() == 'MetroPro EXE'
        self.fill_mode.setItemText(0, '原厂范围（含可补边界孔）' if exe_fill else '限面积内部孔')
        if exe_fill:
            with QSignalBlocker(self.fill_mode):
                self.fill_mode.setCurrentIndex(0)
        self.fill_mode.setEnabled(self.fill.isChecked() and not exe_fill)
        self.fill_max.setToolTip('0 关闭补洞；快速横纵/对角补点先执行，面积上限用于剩余孔洞拟合。' if exe_fill else '最大连通孔洞面积。')
        self.fill_method.setEnabled(self.fill.isChecked())
        self.fill_max.setEnabled(self.fill.isChecked() and self.fill_mode.currentData() == 0)
        enabled = self.filter_mode.currentText() != 'Off'
        physical = self.remove.currentData() in ('Fixed Rad Sphere','Fixed Angle Cone','Variable Angle Cone')
        self.radius.setEnabled(self.remove.currentData() == 'Fixed Rad Sphere')
        self.cone_angle.setEnabled(self.remove.currentData() in ('Fixed Angle Cone','Variable Angle Cone'))
        fft = self.filter_type.currentText() not in ('Average', 'Median', '2 Sigma')
        automatic = self.filter_type.currentText().endswith(' Auto')
        band = self.filter_mode.currentText() in ('Band Pass', 'Band Reject')
        if band and not fft:
            with QSignalBlocker(self.filter_type):
                self.filter_type.setCurrentText('FFT Fixed')
            fft = True
        for index in range(3):
            self.filter_type.model().item(index).setEnabled(not band)
        self.filter_type.setEnabled(enabled)
        self.window.setEnabled(enabled and not fft)
        self.filter_trim.setEnabled(enabled)
        self.filter_trim.setToolTip('频率滤波后裁去 5 层外部边缘；每层重新识别外部背景。' if fft else
            '开启：仅保留窗口内全部点有效的位置；关闭：按有效邻点计算并保留原有效域。')
        for widget in (self.pixel_size, self.cutoff_shape):
            widget.setEnabled(enabled and fft)
        self.pixel_size.setEnabled(physical or (enabled and fft))
        self.cutoff_shape.setEnabled(enabled and self.filter_type.currentText() in ('FFT Fixed', 'FFT Auto'))
        for widget in (self.low_frequency, self.low_wavelength):
            widget.setEnabled(enabled and fft and not automatic and self.filter_mode.currentText() != 'Low Pass')
        for widget in (self.high_frequency, self.high_wavelength):
            widget.setEnabled(enabled and fft and not automatic and self.filter_mode.currentText() != 'High Pass')
        self.fill_max.setEnabled(self.fill.isChecked() and self.fill_mode.currentData() == 0)
        self.spike_height.setEnabled(self.spikes.isChecked())
        self.changed.emit()

    def options(self) -> SurfaceOptions:
        return SurfaceOptions(remove=str(self.remove.currentData()), trim=self.trim.value(),
            trim_mode=self.trim_mode.currentText(), min_area=self.min_area.value(), remove_spikes=self.spikes.isChecked(),
            spike_height=self.spike_height.value(), data_fill=self.fill.isChecked(),
            data_fill_max=self.fill_max.value(), data_fill_mode=self.fill_mode.currentData(),
            data_fill_method=self.fill_method.currentData(), filter_mode=self.filter_mode.currentText(),
            filter_type=self.filter_type.currentText(), window_size=self.window.value(),
            filter_trim=self.filter_trim.isChecked(), pixel_size_um=self.pixel_size.value(),
            low_frequency=self.low_frequency.value(), high_frequency=self.high_frequency.value(),
            cutoff_shape=self.cutoff_shape.currentText(),sphere_radius_mm=self.radius.value(),
            cone_angle_deg=self.cone_angle.value(),
            zernike_center_x=self.zernike.center_x.value(), zernike_center_y=self.zernike.center_y.value(),
            zernike_radius=self.zernike.radius.value(), zernike_terms=self.zernike.terms.currentData(),
            zernike_remove_terms=self.zernike.selected_terms() if self.remove.currentData() == 'Zernike Fringe' else (1,2,3))
