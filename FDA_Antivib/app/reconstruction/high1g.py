"""公共载频相位高度与逐像素级次校正，不执行空间后处理。"""

import numpy as np

from app.reconstruction.spectrum import SpectrumResult


def reconstruct_high1g(front: SpectrumResult, common_k0: float | None = None):
    """光学波数K对应强度相位2Kz，整数高度周期为π/K。"""
    valid = front.valid
    k = (float(front.peak_bin[valid].mean()) * np.pi / (front.frame_count * front.step_um)
         if common_k0 is None else float(common_k0))
    if not np.isfinite(k) or k <= 0:
        raise ValueError("High1G需要有限正公共K0。")
    center_bin = k * front.frame_count * front.step_um / np.pi
    phase = front.intercept + front.slope * center_bin
    phase_height = front.reference_um + phase / (2 * k)
    period = np.pi / k
    delta = phase_height - front.coarse_um
    # 原厂先减去过大的正级次，再补偿负级次；恰好±半周期时保持原值。
    turns = np.where(delta > period / 2, np.ceil((delta - period / 2) / period),
                     np.where(delta < -period / 2, np.floor((delta + period / 2) / period), 0))
    height = np.where(valid, phase_height - turns * period, np.nan)
    return height, {
        'phi0': np.where(valid, phase, np.nan),
        'phi0_map': np.where(valid, phase, np.nan),
        'phase_height_before_order_nm': np.where(valid, phase_height * 1000, np.nan),
        'fringe_order_map': np.where(valid, -turns, np.nan),
        'height_period_nm': period * 1000,
        'postprocessing': 'none',
        'implementation_scope': '公共K0相位细化与半周期级次校正；使用当前实际N帧频谱前端，无空间后处理',
    }
