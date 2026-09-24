"""GFDA 空间相位拟合：七参数包络模型和 13 点 PSI 向量。"""
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
MIN_PIXELS: Final = 16
MAX_CONDITION: Final = 1.e6
MAX_RELATIVE_RESIDUAL: Final = .45
MIN_CONTRAST: Final = .005
PHASE_MARGIN: Final = .01
CONFIDENCE_PIXEL_TARGET: Final = 64


class GfdaError(ValueError):
    """数据不满足 GFDA 可观测性或标定约束，禁止静默转普通 FDA。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"GFDA：{reason}")


@dataclass(frozen=True, slots=True)
class PairEstimate:
    phase_rad: float
    pixel_count: int
    condition: float
    relative_residual: float
    confidence: float


def model_basis(q: FloatArray) -> FloatArray:
    """q 的单位为 rad；正弦负号与原厂七系数模型一致。"""
    cosine, sine = np.cos(q), np.sin(q)
    return np.column_stack((np.ones_like(q), q*q*cosine, q*cosine, cosine,
                            -q*q*sine, -q*sine, -sine))


# m=-6..6，q=m*pi/4；消去直流与一次包络项，不随采样帧数缩放。
PSI_WEIGHTS: Final = np.array([
    [0, -4, -12, -12, 0, 16, 24, 16, 0, -12, -12, -4, 0],
    [-3, -4, 0, 12, 21, 16, 0, -16, -21, -12, 0, 4, 3],
], dtype=float)
PSI_OPERATOR: Final = PSI_WEIGHTS @ model_basis(np.arange(-6, 7)*np.pi/4)


def estimate_pair(heights_um: FloatArray, dc: FloatArray, first: FloatArray,
                  second: FloatArray, group_um: float, carrier: float,
                  phase_half_width: float = np.pi) -> PairEstimate:
    """在同一参考像素组上拟合两帧，返回正向且无跨周歧义的相位步进。"""
    offset = (heights_um-group_um)*2*carrier
    selected = (np.isfinite(offset) & (np.abs(offset) <= phase_half_width)
                & np.isfinite(dc) & (dc > 0) & np.isfinite(first) & np.isfinite(second))
    count = int(selected.sum())
    if count < MIN_PIXELS:
        raise GfdaError(f"参考像素不足（{count}，至少 {MIN_PIXELS}）。")
    q = offset[selected]
    q = q-(q.min()+q.max())/2
    design = model_basis(q)
    # 各像素反射率造成的直流差异先归一化；共同幅值不影响向量夹角。
    signals = np.column_stack((first[selected], second[selected]))/dc[selected, None]-1
    contrast = np.std(signals, axis=0)
    if np.min(contrast) < MIN_CONTRAST:
        raise GfdaError("参考条纹调制度不足。")
    coefficients, _, rank, singular = np.linalg.lstsq(design, signals, rcond=None)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else np.inf
    if rank < design.shape[1] or condition > MAX_CONDITION:
        raise GfdaError("参考像素相位覆盖不足，七参数拟合秩亏或病态。")
    residual = float(np.max(np.sqrt(np.mean((design@coefficients-signals)**2, axis=0))/contrast))
    if residual > MAX_RELATIVE_RESIDUAL:
        raise GfdaError("参考像素不符合共同轴向运动模型，拟合残差过大。")
    vectors = PSI_OPERATOR@coefficients
    norms = np.linalg.norm(vectors, axis=0)
    if not np.isfinite(norms).all() or np.min(norms) < MIN_CONTRAST:
        raise GfdaError("拟合相位向量退化。")
    dot = float(vectors[:, 0]@vectors[:, 1])
    cross = float(vectors[0, 0]*vectors[1, 1]-vectors[1, 0]*vectors[0, 1])
    phase = float(np.arccos(np.clip(dot/np.prod(norms), -1., 1.)))
    # 高度递增坐标中，正向扫描使空间相位递减；保留 acos 幅值并检查方向。
    if cross >= 0 or not PHASE_MARGIN < phase < np.pi-PHASE_MARGIN:
        raise GfdaError("扫描步进反向、接近零或存在半周期歧义。")
    # 这是拟合质量分数，不是测量误差的统计置信概率，也不是 64 帧要求。
    confidence = float((1-residual)*min(1., count/CONFIDENCE_PIXEL_TARGET))
    return PairEstimate(phase, count, condition, residual, confidence)
