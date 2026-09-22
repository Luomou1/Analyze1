"""Micro.app 的 EXE 补洞分支；与 Analysis.Core Polynomial 分开实现。"""
import numpy as np
from scipy import ndimage


def exe_fill(data: np.ndarray, maximum: int, mode: int = 0) -> tuple[np.ndarray, int]:
    """独立浮点实现 0x985410；Max 约束拟合孔洞，快速补点先执行。"""
    from app.core.surface_options import SurfaceAnalysisError
    if mode != 0 or maximum < 0:
        raise SurfaceAnalysisError('MetroPro EXE 补洞只支持原厂范围及非负面积上限。')
    result = data.copy()
    missing_count = int(np.count_nonzero(~np.isfinite(result)))
    if maximum == 0:
        return result, 0
    height, width = result.shape
    # EXE 0x984180/0x984220：先左右均值，再上下均值。相邻缺失点不能
    # 互相启动同一方向的填充，因此每个方向可以批量计算，不改变扫描结果。
    for axis in (1, 0):
        valid = np.isfinite(result)
        pairs = (valid[1:-1, :-2] & valid[1:-1, 2:] if axis == 1 else
                 valid[:-2, 1:-1] & valid[2:, 1:-1])
        yy, xx = np.nonzero(~valid[1:-1, 1:-1] & pairs)
        yy, xx = yy+1, xx+1
        result[yy, xx] = ((result[yy, xx-1]+result[yy, xx+1])/2 if axis == 1 else
                          (result[yy-1, xx]+result[yy+1, xx])/2)
    # 0x984450 仅在恰好一对对角有效时填充；必须保留行优先原位更新。
    for y, x in np.argwhere(~np.isfinite(result[1:-1, 1:-1]))+1:
        values = result[[y-1, y-1, y+1, y+1], [x-1, x+1, x-1, x+1]]
        bits = sum(1 << i for i, v in enumerate(values) if np.isfinite(v))
        if bits in (6, 9):
            result[y, x] = np.nanmean(values)
    labels, count = ndimage.label(~np.isfinite(result))
    sizes = np.bincount(labels.ravel())
    bounds = ndimage.find_objects(labels)
    for label in sorted(range(1, count+1), key=lambda i: -sizes[i]):
        if sizes[label] > maximum:
            continue
        rows, cols = bounds[label-1]
        y0, x0 = max(0, rows.start-1), max(0, cols.start-1)
        # EXE 使用孔洞边长加 4 的非对称邻域，上限停在矩阵最后两行/列之前；
        # 不得套用 Core DLL 的边长加 2，也不能把边界孔一律排除。
        y1 = y0+min(rows.stop-rows.start+4, height-2-y0)
        x1 = x0+min(cols.stop-cols.start+4, width-2-x0)
        if y1 <= y0 or x1 <= x0:
            continue
        region = result[y0:y1, x0:x1]
        yy, xx = np.indices(region.shape, dtype=float)
        valid = np.isfinite(region)
        if not valid.any():
            continue
        design = np.stack((xx*xx, yy*yy, xx*yy, xx, yy, np.ones_like(xx)), axis=-1)
        coef, _, rank, _ = np.linalg.lstsq(design[valid], region[valid], rcond=None)
        iy, ix = np.nonzero(labels[rows, cols] == label)
        iy, ix = iy+rows.start, ix+cols.start
        dy, dx = iy-y0, ix-x0
        prediction = (np.stack((dx*dx, dy*dy, dx*dy, dx, dy, np.ones_like(dx)), axis=-1) @ coef
                      if rank == 6 and sizes[label] > 1 else np.mean(region[valid]))
        result[iy, ix] = prediction
    return result, missing_count-int(np.count_nonzero(~np.isfinite(result)))
