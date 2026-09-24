"""表面显示参数：只修改渲染状态，不修改测量数据。"""
from typing import Callable, Final

import pyvista as pv
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QSlider, QWidget

SURFACE_CMAP: Final = "jet"
DEFAULT_RELIEF: Final = 25


def surface_color_limits(data: np.ndarray) -> tuple[float, float]:
    """二维、三维共用实际高度范围；坐标轴留白不参与颜色归一化。"""
    values = np.asarray(data)[np.isfinite(data)]
    if not values.size:
        return -0.5, 0.5
    lower, upper = float(values.min()), float(values.max())
    if lower == upper:
        return lower - 0.5, upper + 0.5
    return lower, upper


def compact_scalar_bar() -> dict:
    """右上角短色条，只显示端点与单位，保留足够宽度容纳读数。"""
    return {
        "title": "nm", "vertical": True,
        "position_x": 0.86, "position_y": 0.69,
        "width": 0.12, "height": 0.24,
        "n_labels": 2, "fmt": "%.3g",
        "label_font_size": 10, "title_font_size": 10,
        "color": "#e2e8f0", "outline": False,
        "unconstrained_font_size": True,
    }


def surface_geometry(grid: pv.StructuredGrid, values: np.ndarray) -> pv.PolyData:
    """在计算法线前真正移除孔洞面，防止隐藏点的占位高度影响邻域光照。"""
    grid.point_data["height"] = values.ravel(order="F")
    # geometry 过滤器直接排除隐藏单元，避免先复制完整表面再删除孔洞。
    surface = grid.extract_surface(algorithm="geometry")
    return surface.triangulate()


class SurfaceDisplayControls(QWidget):
    """用演员变换调节高度比例，避免每次拖动重新生成网格和法线。"""

    def __init__(self, changed: Callable[[], None], reset_camera: Callable[[], None]) -> None:
        super().__init__()
        self.relief = QSlider(Qt.Orientation.Horizontal)
        self.relief.setRange(5, 100)
        self.relief.setValue(DEFAULT_RELIEF)
        self.relief.setMaximumWidth(180)
        self.relief.setAccessibleName("三维显示高度比例")
        self.relief.setToolTip("高度范围占横向长边的比例，仅影响显示，不改变高度读数或分析结果。")
        self.value_label = QLabel("25%")
        self.lighting = QCheckBox("立体光照")
        self.lighting.setChecked(True)
        reset = QPushButton("重置视角")
        reset.clicked.connect(reset_camera)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.addWidget(QLabel("显示高度"))
        layout.addWidget(self.relief)
        layout.addWidget(self.value_label)
        layout.addWidget(self.lighting)
        layout.addStretch()
        layout.addWidget(reset)
        self.relief.valueChanged.connect(lambda value: self.value_label.setText(f"{value}%"))
        self.relief.valueChanged.connect(changed)
        self.lighting.toggled.connect(changed)

    def apply(self, plotter: pv.Plotter, shape: tuple[int, int]) -> None:
        """保持 XY 像素比例；Z 归一化后独立缩放，色标始终为原始值。"""
        rows, cols = shape
        longest = max(rows - 1, cols - 1, 1)
        scale = (max(cols - 1, 1) / longest, max(rows - 1, 1) / longest,
                 self.relief.value() / 100.0)
        # 某些 PyVista 版本即使 render=False 仍从 set_scale 触发重绘；
        # 合并为调用方的一次重绘，避免隐藏页签抢占 OpenGL 上下文。
        suppressed = plotter.suppress_rendering
        plotter.suppress_rendering = True
        try:
            plotter.set_scale(*scale, reset_camera=False, render=False)
        finally:
            plotter.suppress_rendering = suppressed
        axes = plotter.renderer.cube_axes_actor
        if axes is not None:
            # 仅修改几何边界；PyVista 的 bounds 属性还会覆盖真实高度刻度。
            axes.SetBounds(0, scale[0], 0, scale[1], 0, scale[2])
        for actor in plotter.actors.values():
            if isinstance(actor, pv.Actor):
                actor.prop.lighting = self.lighting.isChecked()
                actor.prop.ambient = 0.25
                actor.prop.diffuse = 0.75
                actor.prop.specular = 0.12
                actor.prop.specular_power = 24.0
        plotter.reset_camera_clipping_range()


def configure_surface_scene(plotter: pv.Plotter) -> None:
    """使用低开销 MSAA 与斜向灯光强调形貌，不对高度做滤波。"""
    plotter.set_background("#080b10")
    plotter.enable_anti_aliasing("msaa", multi_samples=4)
    plotter.remove_all_lights()
    plotter.add_light(pv.Light(position=(1, -2, 3), focal_point=(0.5, 0.5, 0),
                               light_type="camera light", intensity=0.9))
    plotter.add_light(pv.Light(position=(-2, 1, 1), focal_point=(0.5, 0.5, 0),
                               light_type="camera light", intensity=0.25))
