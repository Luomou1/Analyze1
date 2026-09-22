"""圆孔径 Fringe 泽尼克拟合：1 起编号、非 RMS 归一、像素中心坐标。"""
from dataclasses import dataclass
from math import factorial, isfinite

import numpy as np

from app.core.surface_options import SurfaceAnalysisError


@dataclass(frozen=True, slots=True)
class ZernikeResult:
    processed: np.ndarray
    removed: np.ndarray
    coefficients: np.ndarray


def fringe_modes(terms: int) -> tuple[tuple[int, int], ...]:
    """Fringe j=(1+(n+|m|)/2)^2-2|m|+[m<0]；正 m 为 cos，负 m 为 sin。"""
    modes = {}
    for n in range(11):
        for m in range(-n, n+1, 2):
            index = (1+(n+abs(m))//2)**2-2*abs(m)+(m < 0)
            if index <= terms:
                modes[index] = (n, m)
    return tuple(modes[j] for j in range(1, terms+1))


def fringe_basis(x: np.ndarray, y: np.ndarray, terms: int) -> np.ndarray:
    """在归一圆坐标构造基底，R_n^m(1)=1，不混用 Noll/Standard 归一。"""
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    columns = []
    for n, m in fringe_modes(terms):
        order = abs(m)
        radial = np.zeros_like(rho)
        for k in range((n-order)//2+1):
            coefficient = (-1)**k*factorial(n-k)/(factorial(k)*factorial((n+order)//2-k)*factorial((n-order)//2-k))
            radial += coefficient*rho**(n-2*k)
        columns.append(radial*(np.sin(order*theta) if m < 0 else np.cos(order*theta)))
    return np.stack(columns, axis=-1)


def fit_zernike(data: np.ndarray, center_x: float, center_y: float, radius: float,
                terms: int, remove_terms: tuple[int, ...]) -> ZernikeResult:
    """同时拟合前 N 项，仅扣除指定项；孔径外和原无效点保持 NaN。"""
    if not all(isfinite(v) for v in (center_x, center_y, radius)) or radius <= 0:
        raise SurfaceAnalysisError('泽尼克中心须有限，半径须为正的像素数。')
    if not 1 <= terms <= 36 or any(j < 1 or j > terms for j in remove_terms):
        raise SurfaceAnalysisError('Fringe 拟合项数为 1～36，扣除项必须位于拟合范围内。')
    if len(set(remove_terms)) != len(remove_terms):
        raise SurfaceAnalysisError('泽尼克扣除项不能重复。')
    y, x = np.indices(data.shape, dtype=float)
    x, y = (x-center_x)/radius, (y-center_y)/radius
    valid = (x*x+y*y <= 1) & np.isfinite(data)
    if valid.sum() < terms:
        raise SurfaceAnalysisError('孔径内有效点不足，无法拟合泽尼克。')
    design = fringe_basis(x[valid], y[valid], terms)
    coef, _, rank, _ = np.linalg.lstsq(design, data[valid], rcond=None)
    if rank < terms:
        raise SurfaceAnalysisError('孔径内有效点分布退化，无法拟合泽尼克。')
    removed = np.full(data.shape, np.nan)
    indices = np.array(remove_terms, dtype=int)-1
    removed[valid] = design[:, indices]@coef[indices]
    return ZernikeResult(np.where(valid, data-removed, np.nan), removed, coef)
