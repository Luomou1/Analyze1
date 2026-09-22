"""台阶平台评价与单文件可追溯报告，不改变测高使用的共同高度数据。"""
from dataclasses import asdict, dataclass
import json
import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import numpy as np

from app.core.surface_analysis import StepAnalysisResult, StepMeasurement
from app.core.surface_options import SurfaceAnalysisError, SurfaceOptions
from app.core.surface_processing import surface_stats

Reference = Literal['zero', 'mean', 'plane']


@dataclass(frozen=True, slots=True)
class RegionStatistics:
    mean: float
    sa: float
    sq: float
    pv: float
    valid_count: int
    valid_ratio: float
    coefficients: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class StepRegionReport:
    measurement: StepMeasurement
    reference: Reference
    region_one: RegionStatistics
    region_two: RegionStatistics


def _region_stats(data: np.ndarray, bounds: tuple[int, int, int, int], reference: Reference) -> RegionStatistics:
    """参考面 z=a*x+b*y+c，坐标相对各自框选区域左上角，单位 nm/pixel。"""
    left, top, right, bottom = bounds
    region = data[top:bottom, left:right]
    valid = np.isfinite(region)
    values = region[valid]
    if not values.size:
        raise SurfaceAnalysisError('区域没有有效高度。')
    mean = float(np.mean(values))
    coefficients = (0., 0., 0.)
    match reference:
        case 'zero':
            residual = values
        case 'mean':
            coefficients = (0., 0., mean)
            residual = values-mean
        case 'plane':
            yy, xx = np.indices(region.shape, dtype=float)
            design = np.column_stack((xx[valid], yy[valid], np.ones(values.size)))
            coef, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
            if rank < 3:
                raise SurfaceAnalysisError('平台独立平面至少需要三个不共线的有效点。')
            coefficients = (float(coef[0]), float(coef[1]), float(coef[2]))
            residual = values-design@coef
        case _:
            raise SurfaceAnalysisError('未知平台参考面。')
    return RegionStatistics(mean, float(np.mean(np.abs(residual))),
                            float(np.sqrt(np.mean(residual**2))), float(np.ptp(residual)),
                            int(values.size), float(values.size/region.size), coefficients)


def evaluate_regions(data: np.ndarray, measurement: StepMeasurement, reference: Reference = 'mean') -> StepRegionReport:
    """区域去均值或去倾斜只作用于平台指标，不重新计算台阶均值差。"""
    return StepRegionReport(measurement, reference,
                            _region_stats(data, measurement.region_one, reference),
                            _region_stats(data, measurement.region_two, reference))


def measurement_notes(options: SurfaceOptions | None) -> tuple[str, ...]:
    """说明当前处理对于台阶高度和粗糙度解释的影响。"""
    notes = ['全域 Sa/Sq 相对处理后零基准，包含平台高度差；不等同于单平台粗糙度或完整 ISO 25178 评价。']
    if options is not None:
        if options.remove not in ('None', 'Piston', 'Three Points'):
            notes.append('全域去形状可能改变平台高度差；测高请核对同层参考面。')
        if options.filter_mode != 'Off':
            notes.append('滤波可能改变台阶高度或混合边缘；纹理观察结果不能直接替代未滤波测高。')
        if options.data_fill:
            notes.append('补洞生成值参与统计；请避开跨台阶孔洞并检查原始有效域。')
    return tuple(notes)


def export_step_report(path: str | Path, result: StepAnalysisResult, report: StepRegionReport | None) -> None:
    """NPZ 同时保存高度、掩码及 JSON 参数/区域快照，可用 allow_pickle=False 读取。"""
    metadata = {
        'format_version': 1, 'height_unit': 'nm', 'lateral_unit': 'um/pixel',
        'coordinate_convention': 'zero-based (x0,y0,x1,y1), exclusive upper bounds',
        'reference_plane_coefficients': 'z=a*x+b*y+c; region-local pixels; nm',
        'options': None if result.options is None else asdict(result.options),
        'points': result.points, 'three_point': result.three_point,
        'excluded_spikes': result.noise_count,
        'global_statistics': surface_stats(result.processed),
        'processing_order': ['trim', 'spikes', 'min_area', 'filter', 'remove_form', 'data_fill', 'statistics'],
        'original_sha256': hashlib.sha256(np.ascontiguousarray(result.original, dtype='<f8').tobytes()).hexdigest(),
        'matrix_hash_encoding': 'C-order little-endian float64; original_nm',
        'coefficient_convention': 'Fringe 1-based, unnormalized, y down' if result.options is not None and result.options.remove == 'Zernike Fringe' else result.options.remove if result.options is not None else 'unknown',
        'regions': None if report is None else asdict(report),
        'notes': measurement_notes(result.options),
    }
    encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2)
    target = Path(path)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=target.parent, suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, original_nm=result.original, processed_nm=result.processed,
                                fitted_surface_nm=result.fitted_surface,
                                valid_mask=np.isfinite(result.processed),
                                original_valid_mask=np.isfinite(result.original),
                                coefficients=result.coefficients, report_json=np.array(encoded))
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
