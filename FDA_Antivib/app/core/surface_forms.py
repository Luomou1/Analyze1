"""球面几何距离拟合与可倾斜锥面高度拟合；所有长度先统一为 μm。"""
import numpy as np
from scipy.optimize import least_squares
from app.core.surface_options import SurfaceAnalysisError, SurfaceOptions
from app.core.surface_operators import remove_form


def _cone_height(x: np.ndarray, y: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """按 ConeFitter 的竖直射线二次方程求交，保留凸凹根分支。"""
    dx, dy = x-parameters[0], y-parameters[1]
    u, v = -np.tan(parameters[3:5])
    slope = np.tan(parameters[5])
    squared = slope*slope
    a = (u*u+v*v)*squared-1
    projection = u*dx+v*dy
    b = 2*(squared+1)*projection
    c = squared*((v*v+1)*dx*dx+(u*u+1)*dy*dy-2*u*v*dx*dy)-projection**2
    discriminant = b*b-4*a*c
    if abs(a) < np.finfo(float).eps or np.any(discriminant < 0):
        raise SurfaceAnalysisError('锥角与倾斜参数使曲面无法表示为单值高度，请检查数据及角度。')
    return parameters[2]-(-b+np.sign(slope)*np.sqrt(discriminant))/(2*a)


def remove_physical_form(data: np.ndarray, options: SurfaceOptions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """有效实测点参加拟合；球面用径向误差，锥面用竖直高度误差。"""
    valid = np.isfinite(data)
    if valid.sum() < 6:
        raise SurfaceAnalysisError('高级曲面拟合至少需要 6 个有效点。')
    y, x = np.indices(data.shape,dtype=float)
    x *= options.pixel_size_um
    y *= options.pixel_size_um
    heights = data/1000
    xv, yv, zv = x[valid], y[valid], heights[valid]
    if np.linalg.matrix_rank(np.column_stack((xv,yv,np.ones_like(xv)))) < 3:
        raise SurfaceAnalysisError('有效点分布退化，无法拟合高级曲面。')
    if options.remove == 'Fixed Rad Sphere':
        radius = abs(options.sphere_radius_mm)*1000
        points = np.column_stack((xv,yv,zv))
        fit = least_squares(lambda center: np.linalg.norm(points-center,axis=1)-radius,
                            np.zeros(3),xtol=1e-12,ftol=1e-12,gtol=1e-12)
        center = fit.x
        domain = radius**2-(x-center[0])**2-(y-center[1])**2
        if np.any(domain[valid] < 0):
            raise SurfaceAnalysisError('给定球面半径无法覆盖有效测量域。')
        sign = 1 if center[2] <= np.mean(zv) else -1
        fitted = center[2]+sign*np.sqrt(np.maximum(domain,0))
        coefficients = np.r_[center,sign*radius]
    else:
        fixed = options.remove == 'Fixed Angle Cone'
        if fixed and abs(options.cone_angle_deg) == 180:
            return remove_form(data,'Plane')
        design = np.column_stack((np.ones_like(xv),xv,yv,xv*xv+yv*yv))
        polynomial, _, rank, _ = np.linalg.lstsq(design,zv,rcond=None)
        if rank < 4 or abs(polynomial[3]) <= np.finfo(float).eps:
            raise SurfaceAnalysisError('有效区域退化或曲率不足，无法确定锥面顶点。')
        cx, cy = -.5*polynomial[1:3]/polynomial[3]
        cz = polynomial @ np.array([1,cx,cy,cx*cx+cy*cy])
        angle = options.cone_angle_deg
        gamma = (-np.sign(angle)*np.deg2rad(90-abs(angle)/2)
                 if angle and abs(angle) < 180 else np.sign(polynomial[3])*np.pi/4)
        guess = np.array([cx,cy,cz,0,0,gamma])

        def errors(parameters: np.ndarray) -> np.ndarray:
            full = np.r_[parameters,gamma] if fixed else parameters
            return _cone_height(xv,yv,full)-zv

        fit = least_squares(errors,guess[:5] if fixed else guess,
                            xtol=1e-12,ftol=1e-12,gtol=1e-12)
        coefficients = np.r_[fit.x,gamma] if fixed else fit.x
        fitted = _cone_height(x,y,coefficients)
    if not fit.success or not np.all(np.isfinite(fitted[valid])):
        raise SurfaceAnalysisError('高级曲面拟合未收敛，请检查参数及有效区域。')
    fitted_nm = fitted*1000
    return data-fitted_nm, fitted_nm, coefficients
