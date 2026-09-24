"""GFDA 像素诊断使用正式运行快照；修改界面参数不会改变已有结果的解释。"""
from collections.abc import Mapping
from typing import TypedDict

import numpy as np

from app.reconstruction.gfda_phase import GfdaError
from app.reconstruction.gfda_spectrum import fit_spectrum, nonuniform_spectrum


class PixelDiagnostic(TypedDict):
    x: int
    y: int
    signal_x: np.ndarray
    signal_raw_y: np.ndarray
    signal_dc_y: np.ndarray
    window_y: np.ndarray
    signal_windowed_y: np.ndarray
    k_x: np.ndarray
    amplitude_y: np.ndarray
    k0_x: float
    k0_y: float
    phase_original_y: np.ndarray
    phase_raw_y: np.ndarray
    phase_unwrapped_y: np.ndarray
    fit_point_count: int
    fit_mask_k_x: np.ndarray
    fit_mask_phase_y: np.ndarray
    fit_k_x: np.ndarray
    fit_phase_y: np.ndarray
    fit_valid: bool
    fft_length: int
    k0_index: int
    reference_origin_um: float


def pixel_diagnostics(cube: np.ndarray, x: int, y: int, result: Mapping) -> PixelDiagnostic:
    """返回现有像素窗口所需字段，频谱和拟合使用 GFDA 正式坐标与频点约定。"""
    raw = np.asarray(cube[y, x], dtype=float)
    n = len(raw)
    positions = np.asarray(result['scan_positions_used_um'], dtype=float)
    if len(positions) != n:
        raise GfdaError("诊断快照与干涉帧数不一致。")
    step = float(result['gfda_nominal_step_um'])
    maximum = float(result['gfda_maximum'])
    radius = int(result['gfda_window_size'])
    bins = np.arange(2, n//2, dtype=np.int64)
    all_bins = np.arange(n//2+1, dtype=np.int64)
    k = all_bins*np.pi/(n*step)
    reference = float(np.asarray(result['reference_origin_um'])[y, x])
    signal = raw-np.median(raw)
    original = nonuniform_spectrum(signal[None], positions-positions[0], all_bins, step)[0]
    spectrum = original*np.exp(-2j*k*(reference-positions[0]))
    fit = fit_spectrum(spectrum[None, bins], bins, 16*(.07*maximum)**2, radius)
    valid = bool(np.asarray(result['valid_mask'])[y, x])
    good = np.isfinite(fit.unwrapped[0]) & valid
    unwrapped = np.full(len(k), np.nan)
    unwrapped[bins[good]] = fit.unwrapped[0, good]
    amplitude = np.abs(spectrum)*2/n
    amplitude[0] = 0.
    wrapped, original_phase = np.angle(spectrum), np.angle(original)
    wrapped[0], original_phase[0] = np.nan, np.nan
    wrapped[np.abs(spectrum) == 0], original_phase[np.abs(original) == 0] = np.nan, np.nan
    peak = float(np.asarray(result['peak_bin_map'])[y, x])
    k0 = peak*np.pi/(n*step) if valid else np.nan
    slope = float(np.asarray(result['fit_slope_map'])[y, x])
    intercept = float(np.asarray(result['fit_intercept_map'])[y, x])
    return {'x': x, 'y': y, 'signal_x': positions, 'signal_raw_y': raw,
            'signal_dc_y': signal, 'window_y': np.ones(n), 'signal_windowed_y': signal,
            'k_x': k, 'amplitude_y': amplitude, 'k0_x': k0,
            'k0_y': float(np.interp(k0, k, amplitude)) if valid else np.nan,
            'phase_original_y': original_phase, 'phase_raw_y': wrapped,
            'phase_unwrapped_y': unwrapped, 'fit_point_count': int(good.sum()),
            'fit_mask_k_x': k[bins[good]], 'fit_mask_phase_y': fit.unwrapped[0, good],
            'fit_k_x': k[bins[good]], 'fit_phase_y': intercept+slope*bins[good],
            'fit_valid': valid, 'fft_length': n, 'k0_index': int(round(peak)) if valid else -1,
            'reference_origin_um': reference}
