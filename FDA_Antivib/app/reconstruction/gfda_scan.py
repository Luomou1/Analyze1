"""从空间条纹估计共同扫描运动，保持原厂参考组偏移标定的方向约定。"""
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from app.reconstruction.gfda_calibration import Calibration, load_calibration
from app.reconstruction.gfda_phase import GfdaError, MIN_PIXELS, PairEstimate, estimate_pair

FloatArray = NDArray[np.float64]
MAX_REFERENCE_PIXELS: Final = 50_000
BOUNDARY_PHASE_SPAN: Final = 5*np.pi
MIN_CALIBRATION_PAIRS: Final = 5
NOMINAL_PHASE_MARGIN: Final = .05


@dataclass(frozen=True, slots=True)
class ScanEstimate:
    raw_positions: FloatArray
    positions: FloatArray
    phase: FloatArray
    confidence: FloatArray
    counts: NDArray[np.int64]
    residual: FloatArray
    reference_range: tuple[int, int]
    calibration: Calibration


def estimate_scan(cube: FloatArray, initial_heights: FloatArray, valid: NDArray[np.bool_],
                  step_um: float, carrier: float, start_um: float,
                  calibration_path: str | Path | None = None,
                  progress: Callable[[int], None] | None = None) -> ScanEstimate:
    """坐标含未观测尾部的名义延拓；confidence=0 明确标识，调用者检查信号覆盖。"""
    n = cube.shape[-1]
    selected = np.flatnonzero(valid.ravel() & np.isfinite(initial_heights.ravel()))
    if len(selected) < MIN_PIXELS:
        raise GfdaError("有效参考像素不足。")
    selected = selected[::max(1, int(np.ceil(len(selected)/MAX_REFERENCE_PIXELS)))]
    signals = cube.reshape(-1, n)[selected].astype(float)
    heights = initial_heights.ravel()[selected]-start_um
    dc = np.median(signals, axis=1)
    # 覆盖 13 点 PSI 评价的完整 [-3π/2,3π/2]，避免从窄相位带外推二次包络。
    loaded_calibration = (load_calibration(calibration_path, step_um, carrier)
                          if calibration_path is not None else None)
    half_width = loaded_calibration.phase_half_width if loaded_calibration is not None else 1.5*np.pi
    nominal_phase = 2*carrier*step_um
    if not NOMINAL_PHASE_MARGIN < nominal_phase < np.pi-NOMINAL_PHASE_MARGIN:
        raise GfdaError("名义相位步进过小或接近/超过半周期，不支持此采样配置。")
    radius_um = half_width/(2*carrier)
    low = max(0, int(np.ceil((np.min(heights)+radius_um)/step_um-.5)))
    high = min(n-2, int(np.floor((np.max(heights)-radius_um)/step_um-.5)))
    if high-low+1 < (1 if calibration_path is not None else MIN_CALIBRATION_PAIRS):
        raise GfdaError("参考高度/空间相位覆盖不足，无法建立可靠步进标定；请提供含空间条纹的数据。")
    radius = max(1, int(round(BOUNDARY_PHASE_SPAN/nominal_phase)))

    def pair(frame: int, group: int) -> PairEstimate:
        return estimate_pair(heights, dc, signals[:, frame], signals[:, frame+1],
                             (np.clip(group, low, high)+.5)*step_um, carrier, half_width)

    if loaded_calibration is not None:
        calibration = loaded_calibration
        radius = calibration.radius
    else:
        middle = (low+high)//2
        references = range(max(low, middle-radius), min(high, middle+radius)+1)
        ratios: list[list[float]] = [[] for _ in range(2*radius+1)]
        for frame in references:
            if progress:
                progress(int(50*(frame-references.start)/len(references)))
            try:
                baseline = pair(frame, frame).phase_rad
            except GfdaError:
                continue
            for index, offset in enumerate(range(-radius, radius+1)):
                try:
                    ratios[index].append(pair(frame, frame+offset).phase_rad/baseline)
                except GfdaError:
                    continue
        if any(len(values) < MIN_CALIBRATION_PAIRS for values in ratios):
            raise GfdaError("有效参考帧不足以生成边界校正表。")
        coefficients = np.array([np.mean(values) for values in ratios])
        calibration = Calibration(coefficients, step_um, carrier, half_width, '本次扫描内存标定')
        calibration.validate()
    phase = np.full(n-1, nominal_phase)
    confidence, residual = np.zeros(n-1), np.full(n-1, np.nan)
    counts = np.zeros(n-1, dtype=np.int64)
    for frame in range(max(0, low-radius), min(n-2, high+radius)+1):
        if progress:
            progress(50+int(50*(frame-max(0,low-radius))/max(1,min(n-2,high+radius)-max(0,low-radius)+1)))
        try:
            estimate = pair(frame, frame)
        except GfdaError as error:
            if low <= frame <= high:
                raise GfdaError(f"第 {frame+1}–{frame+2} 帧无法估计：{error.reason}") from error
            continue
        offset = frame-int(np.clip(frame, low, high))
        factor = calibration.coefficients[offset+radius] if offset else 1.
        corrected = estimate.phase_rad*factor
        if not 0 < corrected < np.pi:
            raise GfdaError(f"第 {frame+1} 帧边界补偿后步进超出可靠范围。")
        phase[frame], confidence[frame] = corrected, estimate.confidence
        counts[frame], residual[frame] = estimate.pixel_count, estimate.relative_residual
    observed = np.flatnonzero(confidence > 0)
    if np.any(confidence[observed[0]:observed[-1]+1] == 0):
        raise GfdaError("扫描有效区内存在无法观测的步进，不能跨越缺口恢复相对高度。")
    raw = start_um+np.r_[0., np.cumsum(phase/(2*carrier))]
    positions = raw.copy()
    # 独立缓冲区的一次 [1/4,1/2,1/4]，不得原地递推或反复平滑。
    positions[1:-1] = .25*raw[:-2]+.5*raw[1:-1]+.25*raw[2:]
    if np.any(np.diff(positions) <= 0):
        raise GfdaError("恢复坐标非单调，无法执行 GFDA。")
    return ScanEstimate(raw, positions, phase, confidence, counts, residual, (low, high), calibration)
