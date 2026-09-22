"""表面处理及测量结果；保持原始矩阵和各阶段有效域。"""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage

from app.core.surface_options import SurfaceOptions, SurfaceAnalysisError
from app.core.surface_operators import clip_spikes, remove_form, trim_edges, window_filter
from app.core.surface_fft import fft_filter
from app.core.surface_spline import spline_filter
from app.core.surface_filter_controls import resolve_auto, trim_filter_edges
from app.core.surface_forms import remove_physical_form
from app.core.surface_zernike import fit_zernike
from app.core.surface_differential import differential_fill
from app.core.surface_exe_fill import exe_fill


@dataclass(frozen=True, slots=True)
class SurfaceResult:
    original: np.ndarray
    processed: np.ndarray
    fitted_surface: np.ndarray
    coefficients: np.ndarray
    spike_count: int
    filled_count: int
    stats: dict[str, float]
    options: SurfaceOptions


def height_matrix(data: np.ndarray) -> np.ndarray:
    matrix = np.asarray(data, dtype=float).copy()
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise SurfaceAnalysisError('高度数据必须是至少 2×2 的二维矩阵。')
    if np.isinf(matrix).any():
        raise SurfaceAnalysisError('高度矩阵包含无穷值。')
    if not np.any(np.isfinite(matrix)):
        raise SurfaceAnalysisError('高度矩阵没有有效点。')
    return matrix


def surface_stats(data: np.ndarray) -> dict[str, float]:
    """原厂 SumStats：Ra/RMS 相对输入零基准，样本标准差单独计算。"""
    values = data[np.isfinite(data)]
    if values.size == 0:
        raise SurfaceAnalysisError('处理后没有有效点，请检查裁剪、去尖峰和滤波参数。')
    pv = float(np.ptp(values)); sq = float(np.sqrt(np.mean(values**2)))
    return {'sa': float(np.mean(np.abs(values))), 'sq': sq, 'pv': pv,
            'height_range': pv, 'rms': sq, 'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)) if values.size > 1 else 0.,
            'minimum': float(np.min(values)), 'maximum': float(np.max(values)),
            'valid_count': int(values.size), 'valid_ratio': float(values.size/data.size)}


def _fill_holes(data: np.ndarray, maximum: int, mode: int = 0) -> tuple[np.ndarray, int]:
    """DataFiller：0 按面积限内部孔，1 全部内部孔，2 包含边界孔。"""
    if mode not in (0, 1, 2):
        raise SurfaceAnalysisError('未知补洞模式。')
    result = data.copy()
    if mode == 0 and maximum == 0:
        return result, 0
    missing = ~np.isfinite(result)
    # 原厂先填四邻全部有效的孤立点，再按面积从大到小拟合连通孔洞。
    # 四邻相接的缺失点不可能被此步骤启动，因此可同时处理孤立点。
    valid = np.isfinite(result)
    isolated = missing[1:-1, 1:-1] & valid[:-2, 1:-1] & valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
    yy, xx = np.nonzero(isolated)
    yy, xx = yy+1, xx+1
    result[yy, xx] = (result[yy, xx+1]+result[yy, xx-1]+result[yy-1, xx]+result[yy+1, xx])*.25
    filled = len(yy)
    missing = ~np.isfinite(result)
    labels, count = ndimage.label(missing, structure=ndimage.generate_binary_structure(2, 1))
    sizes = np.bincount(labels.ravel())
    bounds = ndimage.find_objects(labels)
    for component in sorted(range(1, count+1), key=lambda item: -sizes[item]):
        area = int(sizes[component])
        if mode == 0 and area > maximum:
            continue
        rows, cols = bounds[component-1]
        if mode != 2 and (rows.start == 0 or cols.start == 0 or rows.stop == result.shape[0] or cols.stop == result.shape[1]):
            continue
        y0, x0 = max(0, rows.start-1), max(0, cols.start-1)
        y1 = min(result.shape[0], y0+rows.stop-rows.start+2)
        x1 = min(result.shape[1], x0+cols.stop-cols.start+2)
        region = result[y0:y1, x0:x1]; valid = np.isfinite(region)
        yy, xx = np.indices(region.shape, dtype=float)
        design = np.stack((xx*xx, yy*yy, xx*yy, xx, yy, np.ones_like(xx)), axis=-1)
        if not valid.any():
            continue
        coef, _, rank, _ = np.linalg.lstsq(design[valid], region[valid], rcond=None)
        # 原厂拟合失败后使用包围框有效数据的均值，不留下伪造的二次系数。
        values = design @ coef if rank == 6 else float(np.mean(region[valid]))
        local_hole = labels[y0:y1, x0:x1] == component
        result[y0:y1, x0:x1] = np.where(local_hole, values, region)
        filled += area
    return result, filled


def process_surface(data: np.ndarray, options: SurfaceOptions, points=()) -> SurfaceResult:
    """PhaseContext 常规路径：裁剪、尖峰、区域筛选、滤波、去形状、补洞、统计。"""
    original = height_matrix(data)
    options = resolve_auto(options, original.shape)
    if options.remove == 'Three Points':
        # 必须先在实测数据上验证参考点，禁止以补洞生成的值建立高度基准。
        remove_form(original, 'Three Points', points)
    filled = 0
    # EXE preZernikeProcess 0x251B80：Trim、Spikes、Min Area、filterData。
    # process 0x252900 随后去形状，并在 0x254D47 调用 Data Fill。
    working = trim_edges(original, options.trim, options.trim_mode)
    spikes = 0
    if options.remove_spikes:
        working, spikes = clip_spikes(working, options.spike_height)
    if options.min_area:
        # RegionFinderConnected：四连通有效点区域，面积恰好等于下限时保留。
        labels, _ = ndimage.label(np.isfinite(working), structure=ndimage.generate_binary_structure(2, 1))
        sizes = np.bincount(labels.ravel())
        keep = (labels != 0) & (sizes[labels] >= options.min_area)
        working = np.where(keep, working, np.nan)
    residual = working
    if options.filter_mode != 'Off':
        if options.filter_type in ('FFT Fixed', 'FFT Auto'):
            residual = fft_filter(residual, options)
        elif options.filter_type in ('Gauss Spline', 'Robust Gauss Spline', 'Gauss Spline Auto', 'Robust Gauss Spline Auto'):
            residual = spline_filter(residual, options)
        else:
            low = window_filter(residual, options.filter_type, options.window_size, options.filter_trim)
            residual = low if options.filter_mode == 'Low Pass' else residual-low
        if options.filter_trim and options.filter_type not in ('Average','Median','2 Sigma'):
            residual = trim_filter_edges(residual)
    if options.remove == 'Zernike Fringe':
        zernike = fit_zernike(residual, options.zernike_center_x, options.zernike_center_y,
                              options.zernike_radius, options.zernike_terms, options.zernike_remove_terms)
        residual, fitted, coefficients = zernike.processed, zernike.removed, zernike.coefficients
    elif options.remove in ('Fixed Rad Sphere','Fixed Angle Cone','Variable Angle Cone'):
        residual, fitted, coefficients = remove_physical_form(residual, options)
    else:
        residual, fitted, coefficients = remove_form(residual, options.remove, points)
    if options.data_fill:
        filler = {'MetroPro EXE': exe_fill, 'Differential': differential_fill,
                  'Polynomial': _fill_holes}[options.data_fill_method]
        residual, filled = filler(residual, options.data_fill_max, options.data_fill_mode)
        if options.remove == 'Zernike Fringe':
            # 圆孔径外不是待补测量孔，含边界模式也不得扩张用户选择的孔径。
            before_mask = np.isfinite(residual)
            yy, xx = np.indices(residual.shape)
            aperture = (xx-options.zernike_center_x)**2+(yy-options.zernike_center_y)**2 <= options.zernike_radius**2
            residual[~aperture] = np.nan
            filled -= int(np.count_nonzero(before_mask & ~aperture))
    return SurfaceResult(original, residual, fitted, coefficients, spikes, filled,
                         surface_stats(residual), options)
