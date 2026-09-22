"""原厂普通 Gaussian spline：自然边界五对角系统及逐轴求解。"""
from typing import Final

import numpy as np
from scipy.linalg import solveh_banded

from app.core.surface_operators import remove_form
from app.core.surface_options import SurfaceAnalysisError, SurfaceOptions

DEFAULT_TENSION: Final = 0.625242
DEFAULT_BEATON: Final = 4.447806654
MAX_ITERATIONS: Final = 25
RELATIVE_TOLERANCE: Final = 0.001


def _smooth_axis(values: np.ndarray, frequency: float, weights: np.ndarray | None = None) -> np.ndarray:
    """沿首轴求解，三行带状存储避免构造像素数平方的稠密矩阵。"""
    length = values.shape[0]
    if length < 4:
        raise SurfaceAnalysisError('Gauss Spline 每个方向至少需要 4 个像素。')
    scale = 1 / (2*np.sin(np.pi*frequency))**2
    first = DEFAULT_TENSION * scale
    second = (1-DEFAULT_TENSION) * scale**2
    diagonal = np.full(length, 1+2*first+6*second)
    diagonal[[0, -1]] = 1+first+second
    diagonal[[1, -2]] = 1+2*first+5*second
    adjacent = np.full(length-1, -first-4*second)
    adjacent[[0, -1]] = -first-2*second
    bands = np.zeros((3, length))
    bands[0] = diagonal
    bands[1, :-1] = adjacent
    bands[2, :-2] = second
    if weights is not None:
        result = np.empty_like(values)
        for column in range(values.shape[1]):
            # 只有行方向使用 Beaton 权重；列方向仍求解普通样条。
            bands[0] = diagonal-1+weights[:, column]
            try:
                result[:, column] = solveh_banded(bands, weights[:, column]*values[:, column], lower=True)
            except np.linalg.LinAlgError as exc:
                raise SurfaceAnalysisError('鲁棒样条权重导致系统退化，请调整截止频率或有效区域。') from exc
        return result
    return solveh_banded(bands, values, lower=True)


def spline_filter(data: np.ndarray, options: SurfaceOptions) -> np.ndarray:
    """二次去形状后补零求解，输出时恢复原有效域。"""
    residual, fitted, _ = remove_form(data, 'Cylinder')
    valid = np.isfinite(data)
    values = np.where(valid, residual, 0)

    def low_pass(frequency_per_mm: float, robust: bool = False) -> np.ndarray:
        frequency = frequency_per_mm * options.pixel_size_um / 1000
        rows = _smooth_axis(values.T, frequency).T
        estimate = _smooth_axis(rows, frequency)
        if robust:
            # 原厂尺度为各行绝对残差中位数的中位数，不是全矩阵中位数。
            cutoff = DEFAULT_BEATON*np.median(np.median(np.abs(values-estimate), axis=1))
            for _ in range(MAX_ITERATIONS):
                if cutoff**2 < np.finfo(float).eps:
                    # 原厂将权重置零；去形状后的近零残差按零解处理。
                    estimate = np.zeros_like(values)
                    break
                previous = cutoff
                weights = np.maximum(1-((values-estimate)/cutoff)**2, 0)**2
                rows = _smooth_axis(values.T, frequency, weights.T).T
                estimate = _smooth_axis(rows, frequency)
                cutoff = DEFAULT_BEATON*np.median(np.median(np.abs(values-estimate), axis=1))
                if cutoff < np.finfo(float).eps or abs(cutoff-previous)/cutoff <= RELATIVE_TOLERANCE:
                    break
        return estimate + fitted

    robust = options.filter_type.startswith('Robust Gauss Spline')
    # 保持既有 str 参数接口；未知模式仍由业务异常明确拒绝。
    match options.filter_mode:  # noqa: MATCH_OK
        case 'Low Pass':
            result = low_pass(options.high_frequency, robust)
        case 'High Pass':
            result = data-low_pass(options.low_frequency, robust)
        case 'Band Pass':
            result = low_pass(options.high_frequency, robust)-low_pass(options.low_frequency)
        case 'Band Reject':
            result = data-low_pass(options.high_frequency, robust)+low_pass(options.low_frequency)
        case _:
            raise SurfaceAnalysisError('普通样条仅开放低通、高通和带通。')
    return np.where(valid, result, np.nan)
