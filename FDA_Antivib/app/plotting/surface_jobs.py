"""后台准备独占的 VTK 网格，Qt/OpenGL 对象只在界面线程访问。"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyvista as pv
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


@dataclass
class PreparedSurface:
    values: np.ndarray
    x: np.ndarray
    y: np.ndarray
    bounds: tuple[float, float]
    mesh: pv.PolyData


def prepare_surface(data: np.ndarray, bounds=None, padded: bool = False) -> PreparedSurface:
    from app.plotting.preview import build_normalized_surface_grid
    from app.plotting.surface_style import surface_geometry
    if padded and bounds is None:
        finite = data[np.isfinite(data)]
        if finite.size:
            center = float(np.median(finite))
            radius = max(center - finite.min(), finite.max() - center)
            half = max(radius * 1.1, 0.5)
            bounds = (center - half, center + half)
    grid, values, x, y, resolved = build_normalized_surface_grid(data, z_bounds=bounds)
    mesh = surface_geometry(grid, values)
    if mesh.n_cells:
        mesh.compute_normals(point_normals=True, cell_normals=False,
                             split_vertices=False, inplace=True)
    return PreparedSurface(values, x, y, resolved, mesh)


class _Signals(QObject):
    done = Signal(int, object, str)


class _Build(QRunnable):
    def __init__(self, generation: int, operation: Callable, signals: _Signals):
        super().__init__()
        self.generation, self.operation, self.signals = generation, operation, signals

    def run(self) -> None:
        try:
            result = self.operation()
        except Exception as error:
            # 线程入口将异常送回界面，不能让准备失败变成永久等待。
            self.signals.done.emit(self.generation, None, str(error))
        else:
            self.signals.done.emit(self.generation, result, "")


class SurfaceJobs(QObject):
    """只交付最新请求；关闭或新请求会使在途旧结果失效。"""
    def __init__(self, parent, failed: Callable[[str], None]):
        super().__init__(parent)
        self.generation = 0
        self.failed = failed
        self.callback = None
        self._signals = {}

    def submit(self, operation: Callable, callback: Callable) -> None:
        self.generation += 1
        self.callback = callback
        signals = _Signals()
        signals.done.connect(self._done)
        self._signals[self.generation] = signals
        QThreadPool.globalInstance().start(_Build(self.generation, operation, signals))

    def cancel(self) -> None:
        self.generation += 1
        self.callback = None

    @Slot(int, object, str)
    def _done(self, generation, result, error) -> None:
        self._signals.pop(generation, None)
        if generation != self.generation or self.callback is None:
            return
        callback, self.callback = self.callback, None
        if error:
            self.failed(f"三维准备失败：{error}")
        else:
            callback(result)
