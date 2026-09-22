"""显式圆孔径与 Fringe 选择性扣除控件。"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QLineEdit

from app.core.surface_options import SurfaceAnalysisError


class ZernikeControls(QGroupBox):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__('泽尼克圆孔径（Fringe）')
        layout = QFormLayout(self)
        self.center_x = self._coordinate()
        self.center_y = self._coordinate()
        self.radius = self._coordinate()
        self.radius.setMinimum(0)
        self.radius.setSpecialValueText('请填写孔径半径')
        layout.addRow('中心 X（列，0 起）', self.center_x)
        layout.addRow('中心 Y（行，0 起）', self.center_y)
        layout.addRow('半径', self.radius)
        self.terms = QComboBox()
        for count in (4, 9, 16, 25, 36):
            self.terms.addItem(str(count), count)
        self.terms.setCurrentIndex(1)
        layout.addRow('拟合项数', self.terms)
        self.remove_terms = QLineEdit('1,2,3')
        self.remove_terms.setPlaceholderText('逗号分隔；留空只拟合')
        layout.addRow('扣除项（1 起编号）', self.remove_terms)
        label = QLabel('非 RMS 归一 Fringe；1 偏置、2/3 倾斜、4 离焦。Y 向下。系数单位 nm，孔径外不参与评价。')
        label.setWordWrap(True)
        layout.addRow(label)
        self.terms.currentIndexChanged.connect(self.changed)
        self.remove_terms.textChanged.connect(self.changed)

    def _coordinate(self) -> QDoubleSpinBox:
        field = QDoubleSpinBox()
        field.setRange(-1e6, 1e6)
        field.setDecimals(3)
        field.setSuffix(' pixel')
        field.setKeyboardTracking(False)
        field.valueChanged.connect(self.changed)
        return field

    def selected_terms(self) -> tuple[int, ...]:
        text = self.remove_terms.text().strip()
        if not text:
            return ()
        try:
            return tuple(int(item.strip()) for item in text.replace('，', ',').split(','))
        except ValueError as exc:
            raise SurfaceAnalysisError('泽尼克扣除项请填写逗号分隔的整数编号。') from exc
