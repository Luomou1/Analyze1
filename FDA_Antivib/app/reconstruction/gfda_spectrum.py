"""以实际扫描坐标直接积分；频率网格和实际采样位置相互独立。"""
from dataclasses import dataclass
from collections.abc import Callable
from typing import Final

import numpy as np
from numpy.typing import NDArray

from app.reconstruction.gfda_phase import GfdaError
from app.reconstruction.spectrum import SpectrumResult

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BLOCK_SIZE: Final = 512


def quadrature(positions: FloatArray, step_um: float) -> FloatArray:
    """原厂内部中心差分、端点单侧间距；按名义步长归一化到 DFT 幅值。"""
    if (positions.ndim != 1 or len(positions) < 3 or not np.isfinite(positions).all()
            or not np.isfinite(step_um) or step_um <= 0 or np.any(np.diff(positions) <= 0)):
        raise GfdaError("扫描坐标必须有限、严格递增，且名义步长为正。")
    return np.r_[positions[1]-positions[0], (positions[2:]-positions[:-2])/2,
                 positions[-1]-positions[-2]]/step_um


def nonuniform_spectrum(signals: FloatArray, positions: FloatArray,
                        bins: NDArray[np.int64], step_um: float) -> ComplexArray:
    """正指数变换；均匀坐标时等于 conj(rfft)，不循环移位或重新插值强度。"""
    weights = quadrature(positions, step_um)
    if signals.shape[-1] != len(positions):
        raise GfdaError("干涉帧数与扫描坐标长度不一致。")
    omega = 2*np.pi*np.asarray(bins)/(len(positions)*step_um)
    kernel = np.exp(1j*omega[:, None]*positions[None, :])*weights
    return signals@kernel.T


@dataclass(frozen=True, slots=True)
class SpectrumFit:
    slope: FloatArray
    intercept: FloatArray
    peak: FloatArray
    valid: NDArray[np.bool_]
    left: NDArray[np.int64]
    right: NDArray[np.int64]
    unwrapped: FloatArray
    power: FloatArray


def fit_spectrum(spectra: ComplexArray, bins: NDArray[np.int64],
                 threshold: float, radius: int) -> SpectrumFit:
    """在载频两侧连续高信噪比频点上拟合；不附加均匀扫描的 (-1)^bin 相位。"""
    power = np.abs(spectra)**2
    peaks = np.argmax(power, axis=1)
    rows = np.arange(len(power))
    good = np.isfinite(power) & (power > threshold)
    left, right = peaks.copy(), peaks.copy()
    for direction, edge in ((-1, left), (1, right)):
        alive = good[rows, peaks].copy()
        for distance in range(1, radius+1):
            col = peaks+direction*distance
            alive &= (col >= 0) & (col < len(bins))
            active = rows[alive]
            alive[active] &= good[active, col[active]]
            edge[alive] = col[alive]
    valid = (left < peaks) & (peaks < right) & (right-left >= 2)
    unwrapped = np.full_like(power, np.nan)
    slope, intercept, fractional = (np.full(len(power), np.nan) for _ in range(3))
    for row in rows[valid]:
        lo, hi = left[row], right[row]+1
        x = bins[lo:hi].astype(float)
        phase = np.unwrap(np.angle(spectra[row, lo:hi]))
        weights = power[row, lo:hi]
        center = np.average(x, weights=weights)
        phase_center = np.average(phase, weights=weights)
        denominator = np.sum(weights*(x-center)**2)
        slope[row] = np.sum(weights*(x-center)*(phase-phase_center))/denominator
        intercept[row] = phase_center-slope[row]*center
        unwrapped[row, lo:hi] = phase
        col = peaks[row]
        curvature = 2*(power[row, col-1]+power[row, col+1]-2*power[row, col])
        fractional[row] = bins[col]+(power[row, col-1]-power[row, col+1])/curvature
    return SpectrumFit(slope, intercept, fractional, valid, bins[left], bins[right], unwrapped, power)


def reconstruct_nonuniform(cube: FloatArray, positions: FloatArray, step_um: float,
                           maximum: float = 4095, window_size: int = 3,
                           progress: Callable[[int], None] | None = None) -> SpectrumResult:
    """校正频谱转换为现有 High 2G 的 bin 斜率接口，frame_count 仍是真实帧数。"""
    if cube.ndim != 3 or cube.shape[-1] < 10 or not np.isfinite(cube).all():
        raise GfdaError("需要至少 10 帧有限强度数据。")
    if not isinstance(window_size, int) or window_size < 1 or maximum <= 0:
        raise GfdaError("频点窗口和强度上限必须为正。")
    quadrature(positions, step_um)
    n = cube.shape[-1]
    if len(positions) != n:
        raise GfdaError("干涉帧数与扫描坐标长度不一致。")
    flat = cube.reshape(-1, n)
    bins = np.arange(2, n//2, dtype=np.int64)
    omega = 2*np.pi*bins/(n*step_um)
    kernel = np.exp(1j*omega[:, None]*(positions-positions[0]))*quadrature(positions, step_um)
    b, a, peak, power = (np.full(len(flat), np.nan) for _ in range(4))
    refs = np.zeros(len(flat))
    valid = np.zeros(len(flat), dtype=bool)
    left, right = np.zeros(len(flat), dtype=np.int64), np.zeros(len(flat), dtype=np.int64)
    for start in range(0, len(flat), BLOCK_SIZE):
        end = min(start+BLOCK_SIZE, len(flat))
        block = flat[start:end].astype(float)
        signals = block-np.median(block, axis=1, keepdims=True)
        peak_frame = np.argmax(np.abs(signals), axis=1)
        reference = positions[peak_frame]
        spectra = (signals@kernel.T)*np.exp(-1j*(reference-positions[0])[:, None]*omega)
        fit = fit_spectrum(spectra, bins, 16*(.07*maximum)**2, window_size)
        ok = fit.valid & ~np.any(block >= maximum, axis=1)
        valid[start:end] = ok
        b[start:end], a[start:end], peak[start:end] = fit.slope, fit.intercept, fit.peak
        power[start:end] = fit.power.max(axis=1)
        refs[start:end] = reference
        left[start:end], right[start:end] = fit.left, fit.right
        if progress:
            progress(int(100*end/len(flat)))
    b[~valid], a[~valid], peak[~valid] = np.nan, np.nan, np.nan
    coarse = refs+b*n*step_um/(2*np.pi)
    phase = a+b*peak
    shape = cube.shape[:2]
    arrays = [v.reshape(shape) for v in (coarse, phase, b, a, peak, power, valid, left, right, refs)]
    return SpectrumResult(*arrays, n, step_um)
