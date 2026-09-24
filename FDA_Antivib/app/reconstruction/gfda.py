"""GFDA：初始 FDA、共同扫描运动估计、非均匀频谱及 High 2G 适配。"""
from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np

from app.reconstruction.gfda_phase import GfdaError
from app.reconstruction.gfda_scan import estimate_scan
from app.reconstruction.gfda_spectrum import reconstruct_nonuniform
from app.reconstruction.high2g import reconstruct_high2g
from app.reconstruction.spectrum import SpectrumResult, reconstruct_spectrum

MAX_UNOBSERVED_ENERGY: Final = .03


def reconstruct_gfda(cube: np.ndarray, step_um: float, start_um: float,
                     maximum: float, window_size: int, common_k0: float | None,
                     calibration_path: str | Path | None = None,
                     progress: Callable[[int], None] | None = None) -> tuple[SpectrumResult, dict]:
    """载频建立物理尺度；当前不从待恢复运动中推断独立的绝对长度标定。"""
    if not np.isfinite(start_um):
        raise GfdaError("扫描起点必须为有限数值。")
    initial = reconstruct_spectrum(cube, step_um, start_um, maximum, window_size=window_size,
        progress=(lambda value: progress(int(value*20/70))) if progress else None)
    if not initial.valid.any():
        raise GfdaError("初始 FDA 无有效像素。")
    carrier = (float(initial.peak_bin[initial.valid].mean())*np.pi/(cube.shape[-1]*step_um)
               if common_k0 is None else float(common_k0))
    if np.ptp(initial.coarse_um[initial.valid])*2*carrier < 3*np.pi:
        raise GfdaError("参考高度/空间相位覆盖不足，无法估计共同扫描运动。")
    initial_height, _ = reconstruct_high2g(initial, common_k0=carrier)
    if progress:
        progress(20)
    scan = estimate_scan(cube, initial_height, initial.valid, step_um, carrier,
                         start_um, calibration_path,
                         (lambda value: progress(20+int(value*.4))) if progress else None)
    if progress:
        progress(60)
    front = reconstruct_nonuniform(cube, scan.positions, step_um, maximum, window_size,
                                   (lambda value: progress(60+int(value*.2))) if progress else None)
    observed = np.flatnonzero(scan.confidence > 0)
    # 名义延拓只允许发生在包络能量极弱的尾部；不能以未知步进连接有效干涉区。
    flat = cube.reshape(-1, cube.shape[-1])
    fraction_flat = np.ones(len(flat))
    # 逐块检查包络能量，避免为整幅百万像素数据再分配多个三维临时数组。
    for start in range(0, len(flat), 512):
        block = flat[start:start+512].astype(float)
        centered = block-np.median(block, axis=-1, keepdims=True)
        energy = np.sum(centered**2, axis=-1)
        outside = (np.sum(centered[:, :observed[0]+1]**2, axis=-1)
                   + np.sum(centered[:, observed[-1]+1:]**2, axis=-1))
        fraction_flat[start:start+len(block)] = np.divide(outside, energy,
            out=np.ones_like(energy), where=energy > 0)
    fraction = fraction_flat.reshape(cube.shape[:2])
    front.valid[:] &= fraction < MAX_UNOBSERVED_ENERGY
    if not front.valid.any():
        raise GfdaError("可观测扫描区没有覆盖有效干涉包络。")
    for values in (front.coarse_um, front.phase, front.slope, front.intercept, front.peak_bin):
        values[~front.valid] = np.nan
    extras = {
        'gfda_applied': True, 'gfda_scan_k0': carrier,
        'gfda_calibration': scan.calibration,
        'gfda_reference_range': np.asarray(scan.reference_range, dtype=int),
        'gfda_reference_pixel_count': scan.counts,
        'gfda_fit_residual': scan.residual,
        'gfda_unobserved_energy_fraction': fraction,
        'scan_positions_raw_um': start_um+np.arange(cube.shape[-1])*step_um,
        'scan_positions_estimated_raw_um': scan.raw_positions,
        'scan_positions_used_um': scan.positions,
        'scan_step_raw_um': np.diff(scan.raw_positions),
        'scan_step_used_um': np.diff(scan.positions),
        'scan_step_confidence': scan.confidence,
        'scan_phase_estimated_rad': scan.phase,
        'scan_position_correction_um': scan.positions-(start_um+np.arange(cube.shape[-1])*step_um),
        'gfda_initial_height_um': initial_height,
        'gfda_window_size': window_size,
        'gfda_maximum': maximum,
        'gfda_nominal_step_um': step_um,
    }
    if progress:
        progress(80)
    return front, extras
