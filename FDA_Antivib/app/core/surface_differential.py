"""原厂 DifferentialFiller 默认松弛与停止条件的独立实现。"""
import numpy as np
from scipy import ndimage
from typing import Final

from app.core.surface_options import SurfaceAnalysisError

# 原厂没有硬迭代上限；这里仅防止数值停滞导致后台任务无限运行，超限明确报错。
MAX_ITERATIONS: Final = 100000


def differential_fill(data: np.ndarray, maximum: int, mode: int = 0) -> tuple[np.ndarray, int]:
    """四邻 Laplace 松弛；边界反射，容差为边界 PV/孔面积，按最大更新量停止。"""
    if data.ndim != 2 or min(data.shape) < 2 or np.isinf(data).any():
        raise SurfaceAnalysisError('微分补洞需要至少 2×2 且不含无穷值的矩阵。')
    if mode not in (0, 1, 2) or maximum < 0:
        raise SurfaceAnalysisError('微分补洞模式或面积上限无效。')
    result = np.asarray(data, dtype=float).copy()
    valid = np.isfinite(result)
    labels, _ = ndimage.label(~valid)
    filled = 0
    for component, bounds in enumerate(ndimage.find_objects(labels), 1):
        rows, cols = bounds
        area = np.count_nonzero(labels[rows, cols] == component)
        if mode == 0 and area > maximum:
            continue
        if mode != 2 and (rows.start == 0 or cols.start == 0 or rows.stop == data.shape[0] or cols.stop == data.shape[1]):
            continue
        y0, y1 = max(0, rows.start-1), min(data.shape[0], rows.stop+1)
        x0, x1 = max(0, cols.start-1), min(data.shape[1], cols.stop+1)
        region = result[y0:y1, x0:x1]
        hole = labels[y0:y1, x0:x1] == component
        border = ndimage.binary_dilation(hole) & ~hole & np.isfinite(region)
        boundary = region[border]
        if not boundary.size:
            raise SurfaceAnalysisError('微分补洞缺少有效边界。')
        region[hole] = float(np.mean(boundary))
        tolerance = float(np.ptp(boundary)/area)
        if tolerance:
            omega = 1.5 if area < 100 else 1.6 if area < 1000 else 1.7 if area < 10000 else 1.8 if area < 100000 else 1.9
            points = np.argwhere(hole)
            for _ in range(MAX_ITERATIONS):
                largest = 0.
                # 必须使用原厂行优先的原位更新，Jacobi 或红黑迭代会改变停止时的输出。
                for row, col in points:
                    up, down = (row-1 if row else row+1), (row+1 if row+1 < region.shape[0] else row-1)
                    left, right = (col-1 if col else col+1), (col+1 if col+1 < region.shape[1] else col-1)
                    delta = omega*((region[up, col]+region[down, col]+region[row, left]+region[row, right])*.25-region[row, col])
                    if not np.isfinite(delta):
                        raise SurfaceAnalysisError('微分补洞迭代出现非有限值。')
                    region[row, col] += delta
                    largest = max(largest, abs(delta))
                if not np.isfinite(largest):
                    raise SurfaceAnalysisError('微分补洞迭代出现非有限值。')
                if largest <= tolerance:
                    break
            else:
                raise SurfaceAnalysisError('微分补洞未在迭代上限内收敛。')
        filled += int(area)
    return result, filled
