"""二维高度图的局部光照显示，不改变高度数据和色标。"""
import numpy as np


def apply_height_lighting(canvas, image) -> None:
    """用邻域坡度计算漫反射；缺失邻点不参与差分，避免孔洞产生假暗边。"""
    data = np.asarray(image.get_array(), dtype=np.float64)
    valid = np.isfinite(data)
    lower, upper = image.get_clim()
    span = max(upper - lower, np.finfo(float).eps)
    height = np.where(valid, (data - lower) / span, np.nan)
    longest = max(data.shape) - 1
    gradients = []
    for axis in (0, 1):
        before = np.roll(height, 1, axis=axis)
        after = np.roll(height, -1, axis=axis)
        edge = [slice(None), slice(None)]
        edge[axis] = 0
        before[tuple(edge)] = np.nan
        edge[axis] = -1
        after[tuple(edge)] = np.nan
        left = np.isfinite(before) & valid
        right = np.isfinite(after) & valid
        delta = np.where(left, height - before, 0) + np.where(right, after - height, 0)
        count = left.astype(int) + right.astype(int)
        gradients.append(delta / np.maximum(count, 1) * longest * 0.25)
    dy, dx = gradients
    length = np.sqrt(dx * dx + dy * dy + 1)
    # 与三维相同的环境/漫反射系数；固定俯视光向，不随三维相机旋转。
    diffuse = np.maximum(0, (-dx - 2 * dy + 3) / (length * np.sqrt(14)))
    intensity = np.clip(0.25 + 0.75 * diffuse, 0.25, 1)
    shade = np.zeros((*data.shape, 4), dtype=np.float32)
    shade[..., 3] = np.where(valid, 1 - intensity, 0)
    # 保留原始标量图像作为色条和点选数据源；黑色透明层只调节显示明暗。
    image.axes.imshow(shade, origin=image.origin, extent=image.get_extent(),
                      interpolation="nearest", aspect=image.axes.get_aspect(),
                      zorder=image.get_zorder() + 0.1)
