from __future__ import annotations

"""平面与台阶高度矩阵的独立后处理算法。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from app.core.surface_segmentation import _segment_step, _sample_std
from app.core.surface_options import SurfaceOptions
from app.core.surface_processing import process_surface


@dataclass(frozen=True, slots=True)
class PlaneAnalysisResult:
    """平面分析输出。"""

    original: np.ndarray
    calibrated: np.ndarray
    processed: np.ndarray
    fitted_surface: np.ndarray
    coefficients: np.ndarray
    method: str
    outlier_count: int
    noise_count: int
    stats: dict[str, float]
    options: SurfaceOptions | None = None


@dataclass(frozen=True, slots=True)
class StepAnalysisResult:
    """台阶分析输出。"""

    original: np.ndarray
    leveled: np.ndarray
    processed: np.ndarray
    fitted_surface: np.ndarray
    cluster_map: np.ndarray
    coefficients: np.ndarray
    threshold: float
    points: tuple[tuple[int, int], ...]
    three_point: bool
    noise_count: int
    layer_stats: dict[str, dict[str, float]]
    options: SurfaceOptions | None = None


@dataclass(frozen=True, slots=True)
class StepMeasurement:
    """两个矩形区域的台阶高度测量结果。"""

    region_one_mean: float
    region_two_mean: float
    step_height: float
    region_one: tuple[int, int, int, int]
    region_two: tuple[int, int, int, int]


def _as_height_matrix(data: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    matrix = np.asarray(data, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError("高度数据必须是至少 2×2 的二维矩阵。")
    if np.isinf(matrix).any():
        raise ValueError("高度数据包含无穷值，请先检查源文件。")
    return matrix.copy()


def load_height_matrix(path: str | Path, conversion_factor: float = 1.0) -> np.ndarray:
    """读取空白分隔的二维高度文本矩阵并转换到纳米单位。"""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"高度文件不存在：{source}")
    try:
        matrix = np.loadtxt(source, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法读取高度文件：{exc}") from exc
    if not np.isfinite(conversion_factor) or conversion_factor == 0:
        raise ValueError("高度换算系数必须是非零有限值。")
    return _as_height_matrix(matrix) * float(conversion_factor)


def analyze_plane(data: np.ndarray, options: SurfaceOptions = SurfaceOptions()) -> PlaneAnalysisResult:
    """以共用表面算子分析单个高度矩阵。"""
    result = process_surface(data, options)
    return PlaneAnalysisResult(
        original=result.original,
        calibrated=result.processed,
        processed=result.processed,
        fitted_surface=result.fitted_surface,
        coefficients=result.coefficients,
        method=options.remove,
        outlier_count=result.spike_count,
        noise_count=result.spike_count,
        stats=result.stats,
        options=result.options,
    )

def _layer_statistics(data: np.ndarray, cluster_map: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for layer, name in ((1, "low"), (2, "high")):
        values = data[(cluster_map == layer) & np.isfinite(data)]
        if values.size == 0:
            result[name] = {"flatness": float("nan"), "std": float("nan"), "area_ratio": 0.0}
            continue
        result[name] = {
            "flatness": float(np.max(values) - np.min(values)),
            "std": _sample_std(values),
            "area_ratio": float(values.size / data.size),
        }
    return result


def analyze_step(
    data: np.ndarray,
    *,
    points: Iterable[tuple[int, int]] = (),
    options: SurfaceOptions = SurfaceOptions(remove="None"),
) -> StepAnalysisResult:
    """可选三点调平，按有效高度分层供框选测量。"""
    result = process_surface(data, options, points if options.remove == 'Three Points' else ())
    leveled = result.processed
    cluster_map, threshold = _segment_step(leveled)
    layer_stats = _layer_statistics(leveled, cluster_map)
    return StepAnalysisResult(result.original, leveled, result.processed, result.fitted_surface,
                              cluster_map, result.coefficients, threshold,
                              tuple(points), options.remove == 'Three Points',
                              result.spike_count, layer_stats, result.options)

def _clip_region(
    region: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(round(value)) for value in region)
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    rows, cols = shape
    left = max(0, min(left, cols - 1))
    right = max(left + 1, min(right, cols))
    top = max(0, min(top, rows - 1))
    bottom = max(top + 1, min(bottom, rows))
    return left, top, right, bottom


def compute_step_height(
    data: np.ndarray,
    region_one: tuple[int, int, int, int],
    region_two: tuple[int, int, int, int],
) -> StepMeasurement:
    """计算两个矩形区域的均值差，矩形格式为 (x0, y0, x1, y1)。"""
    matrix = _as_height_matrix(data)
    clipped_one = _clip_region(region_one, matrix.shape)
    clipped_two = _clip_region(region_two, matrix.shape)

    def region_mean(region: tuple[int, int, int, int]) -> float:
        left, top, right, bottom = region
        values = matrix[top:bottom, left:right]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("矩形区域为空，请重新框选。")
        return float(np.mean(values))

    mean_one = region_mean(clipped_one)
    mean_two = region_mean(clipped_two)
    return StepMeasurement(
        region_one_mean=mean_one,
        region_two_mean=mean_two,
        step_height=mean_one - mean_two,
        region_one=clipped_one,
        region_two=clipped_two,
    )
