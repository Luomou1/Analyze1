"""MetroPro.exe High 分支辅助运算；整数残差处理。

不能把 DLL FringeOrderAnalysis 模式2作为 EXE High 的替代。
"""

import numpy as np
from numpy.typing import NDArray


def subtract_plane(values: NDArray[np.float64], coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
    """0x140AABD40：每像素先将拟合平面向零截断，再从整数高度中减去。"""
    if values.ndim!=2 or coefficients.shape!=(3,):
        raise ValueError('Expected a 2-D map and three plane coefficients.')
    y,x=np.indices(values.shape)
    plane=coefficients[0]*x+coefficients[1]*y+coefficients[2]
    return np.where(np.isfinite(values),values-np.trunc(plane),np.nan)


def subtract_histogram_mode(values: NDArray[np.float64]) -> tuple[NDArray[np.float64],int]:
    """0x140AABA60：50桶最高频桶的左边界作为偏置，不用均值/中位数替代。"""
    valid=np.isfinite(values)
    if not valid.any():
        raise ValueError('High residual map has no valid data.')
    low=int(values[valid].min());high=int(values[valid].max())
    if low==high:
        return values.copy(),0
    # 原厂常量0xC8AA18为49，最大值落入第49桶。
    scale=np.float32(49)/np.float32(high-low)
    indices=np.trunc((values[valid]-low).astype(np.float32)*scale).astype(np.int64)
    histogram=np.bincount(indices,minlength=50)
    offset=int(np.float32(np.argmax(histogram))/scale+np.float32(low))
    return np.where(valid,values-offset,np.nan),offset


def wrap_high_residual(values: NDArray[np.float64],period: int) -> NDArray[np.float64]:
    """0x140ABCA88：残差超过约0.53周期才移动整周期，保留原厂容差带。"""
    if period<=0:
        raise ValueError('High residual period must be positive.')
    limit=int(np.float32(period)*np.float32(.53))
    result=values.copy()
    above=result>limit
    result[above]-=np.ceil((result[above]-limit)/period)*period
    below=result < -limit
    result[below]+=np.ceil((-limit-result[below])/period)*period
    return result
