"""经原厂小矩阵对照的独立 NumPy/SciPy 表面算子。"""
import numpy as np
from scipy import ndimage

from app.core.surface_options import SurfaceAnalysisError


def remove_form(data: np.ndarray, mode: str, points=()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """以有效点拟合参考面，输出残差；不恢复原始平均高度。"""
    valid = np.isfinite(data)
    if not np.any(valid):
        raise SurfaceAnalysisError('没有有效高度点。')
    y, x = np.indices(data.shape, dtype=float)
    one = np.ones_like(data)
    if mode == 'None':
        return data.copy(), np.zeros_like(data), np.empty(0)
    if mode == 'Three Points':
        if len(points) != 3:
            raise SurfaceAnalysisError('三点调平需要三个同层且不共线的参考点。')
        rows, cols = np.array(points, dtype=int).T
        if np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= data.shape[0]) or np.any(cols >= data.shape[1]):
            raise SurfaceAnalysisError('参考点超出数据范围。')
        if not np.all(valid[rows, cols]):
            raise SurfaceAnalysisError('参考点包含无效高度，不能用于三点调平。')
        design = np.column_stack((cols, rows, np.ones(3)))
        if np.linalg.matrix_rank(design) < 3:
            raise SurfaceAnalysisError('请选取三个不共线的参考点。')
        coef = np.linalg.solve(design, data[rows, cols])
        fitted = coef[0]*x + coef[1]*y + coef[2]
        return data-fitted, fitted, coef
    # 平移/缩放坐标改善条件数；完整多项式空间及残差不随此变换改变。
    x = (x - (data.shape[1]-1)/2) / max(data.shape)
    y = (y - (data.shape[0]-1)/2) / max(data.shape)
    columns = {
        'Piston': (one,), 'Plane': (x, y, one),
        'Sphere': (x*x+y*y, x, y, one),
        'Cylinder': (x*x, y*y, x*y, x, y, one),
    }
    if mode == '4th Order':
        # PolyFitterVari 完整总阶数分支为 15 项，不含 x⁴y⁴ 等总次数大于四的项。
        # 顺序按 y 次数优先、x 次数递增；系数使用本函数中心化归一坐标。
        columns[mode] = tuple(x**i * y**j for j in range(5) for i in range(5-j))
    if mode not in columns:
        raise SurfaceAnalysisError('未知形状去除类型。')
    design = np.stack(columns[mode], axis=-1)
    coef, _, rank, _ = np.linalg.lstsq(design[valid], data[valid], rcond=None)
    if rank < design.shape[-1]:
        raise SurfaceAnalysisError('有效点不足或分布退化，无法拟合所选形状。')
    fitted = design @ coef
    return data-fitted, fitted, coef


def trim_edges(data: np.ndarray, size: int, mode: str) -> np.ndarray:
    """四邻域收缩；Outside 先填充内部孔洞以识别外部连通背景。"""
    if size == 0:
        return data.copy()
    valid = np.isfinite(data)
    domain = ndimage.binary_fill_holes(valid) if mode == 'Outside' else valid
    keep = ndimage.binary_erosion(domain, iterations=size, border_value=0)
    return np.where(keep & valid, data, np.nan)


def clip_spikes(data: np.ndarray, multiplier: float) -> tuple[np.ndarray, int]:
    """原厂 SpikeClipper 默认先扣除二次曲面，再以全局残差 RMS 判尖峰。"""
    residual, _, _ = remove_form(data, 'Cylinder')
    valid = np.isfinite(residual)
    threshold = multiplier * np.sqrt(np.mean(residual[valid]**2))
    if min(data.shape) < 3:
        return data.copy(), 0
    # SpikeClipper 0xE7BF0 在边界内移完整 3×3 窗口，再减去当前点；
    # 截断成 2×3 邻域会漏掉靠近边缘的尖峰影响。
    kernel = np.ones((3, 3))
    centers = np.ix_(np.clip(np.arange(data.shape[0]), 1, data.shape[0]-2),
                     np.clip(np.arange(data.shape[1]), 1, data.shape[1]-2))
    count = ndimage.convolve(valid.astype(float), kernel, mode='constant', cval=0)[centers]-valid
    total = ndimage.convolve(np.where(valid, residual, 0), kernel, mode='constant', cval=0)[centers]-np.where(valid, residual, 0)
    mean = np.divide(total, count, out=np.zeros_like(data), where=count > 0)
    rejected = valid & (count >= 4) & (np.abs(residual-mean) > threshold)
    result = data.copy(); result[rejected] = np.nan
    return result, int(rejected.sum())


def window_filter(data: np.ndarray, method: str, size: int, trim: bool) -> np.ndarray:
    """有效点窗口统计；严格窗口时丢弃接触孔洞/图外的点，否则截断归一。"""
    valid = np.isfinite(data)
    kernel = np.ones((size, size))
    count = ndimage.convolve(valid.astype(float), kernel, mode='constant', cval=0)
    total = ndimage.convolve(np.where(valid, data, 0), kernel, mode='constant', cval=0)
    mean = np.divide(total, count, out=np.full_like(data, np.nan), where=count > 0)
    if method == 'Average':
        filtered = mean
    elif method in ('Median', '2 Sigma'):
        def reduce_window(values: np.ndarray) -> float:
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return np.nan
            if method == 'Median':
                return float(np.median(finite))
            center = np.mean(finite)
            sigma = np.sqrt(np.mean((finite-center)**2))
            keep = finite[np.abs(finite-center) <= 2*sigma]
            return float(np.mean(keep)) if keep.size else np.nan
        filtered = ndimage.generic_filter(data, reduce_window, size=size, mode='constant', cval=np.nan)
    else:
        raise SurfaceAnalysisError('该滤波算法尚未完成原厂对照，未执行替代滤波。')
    keep = valid & (count == size*size if trim else count > 0)
    return np.where(keep, filtered, np.nan)
