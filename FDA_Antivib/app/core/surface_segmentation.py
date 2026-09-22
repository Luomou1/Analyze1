"""既有台阶分层展示，仅生成标签，不改变测量高度。"""
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def _matlab_hist(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """复刻 MATLAB legacy hist：分箱边界上的值归入左侧箱。"""
    flattened = np.asarray(values, dtype=np.float64).ravel()
    minimum = float(np.min(flattened))
    maximum = float(np.max(flattened))
    if minimum == maximum:
        centers = np.full(bins, minimum, dtype=np.float64)
        counts = np.zeros(bins, dtype=np.float64)
        counts[bins // 2] = flattened.size
        return counts, centers

    edges = np.linspace(minimum, maximum, bins + 1, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    assignments = np.searchsorted(edges[1:-1], flattened, side="left")
    counts = np.bincount(assignments, minlength=bins).astype(np.float64)
    return counts, centers


def _matlab_smooth_three(values: np.ndarray) -> np.ndarray:
    """复刻 smooth(values, 3) 的三点移动平均及端点行为。"""
    source = np.asarray(values, dtype=np.float64)
    smoothed = source.copy()
    if source.size >= 3:
        smoothed[1:-1] = (source[:-2] + source[1:-1] + source[2:]) / 3.0
    return smoothed


def _sample_std(values: np.ndarray) -> float:
    source = np.asarray(values, dtype=np.float64)
    if source.size <= 1:
        return 0.0
    return float(np.std(source, ddof=1))


def _matlab_binary_open_close(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """复刻 imclose(imopen(mask, se), se) 的扩展域边界行为。"""
    opened = ndimage.binary_erosion(mask, structure, border_value=1)
    opened = ndimage.binary_dilation(opened, structure, border_value=0)

    row_margin = structure.shape[0] // 2
    col_margin = structure.shape[1] // 2
    padded = np.pad(
        opened,
        ((row_margin, row_margin), (col_margin, col_margin)),
        mode="constant",
        constant_values=False,
    )
    closed = ndimage.binary_dilation(padded, structure, border_value=0)
    closed = ndimage.binary_erosion(closed, structure, border_value=0)
    return closed[
        row_margin : row_margin + mask.shape[0],
        col_margin : col_margin + mask.shape[1],
    ]


def _segment_step(data: np.ndarray) -> tuple[np.ndarray, float]:
    values = data[np.isfinite(data)]
    counts, centers = _matlab_hist(values, bins=50)
    smoothed = _matlab_smooth_three(counts)
    minimum_height = float(np.max(smoothed) * 0.05) if smoothed.size else 0.0
    peak_indices, properties = find_peaks(smoothed, height=minimum_height, distance=5)

    if peak_indices.size >= 2:
        order = np.argsort(properties["peak_heights"])[::-1][:2]
        selected = peak_indices[order]
        selected_heights = properties["peak_heights"][order]
        height_one, height_two = centers[selected]
        peak_one, peak_two = selected_heights
        if height_one < height_two:
            height_one, height_two = height_two, height_one
        threshold = float((height_one * peak_one + height_two * peak_two) / (peak_one + peak_two))
        if abs(float(height_one - height_two)) < float(np.ptp(values)) * 0.1:
            threshold = float(np.mean(values))
    else:
        threshold = float((np.percentile(values, 30) + np.percentile(values, 70)) / 2.0)

    high_mask = data > threshold
    low_mask = ~high_mask
    total = data.size
    if np.count_nonzero(high_mask) / total < 0.05 or np.count_nonzero(low_mask) / total < 0.05:
        threshold = float(np.median(values))
        high_mask = data > threshold
        low_mask = ~high_mask

    structure = _disk(2)
    high_cleaned = _matlab_binary_open_close(high_mask, structure)
    low_cleaned = _matlab_binary_open_close(low_mask, structure)
    overlap = high_cleaned & low_cleaned
    low_cleaned &= ~overlap
    unassigned = ~(high_cleaned | low_cleaned)
    high_cleaned[unassigned & (data > threshold)] = True
    low_cleaned[unassigned & (data <= threshold)] = True

    if not np.any(high_cleaned) or not np.any(low_cleaned):
        high_cleaned = data > threshold
        low_cleaned = ~high_cleaned

    cluster_map = np.ones(data.shape, dtype=np.uint8)
    cluster_map[high_cleaned] = 2
    cluster_map[low_cleaned] = 1
    cluster_map[~np.isfinite(data)] = 0
    return cluster_map, threshold


