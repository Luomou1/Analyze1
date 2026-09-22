"""主程序 FilterMgr 的自动截止和频率滤波裁边规则。"""
from dataclasses import replace
from typing import Final
import numpy as np
from scipy import ndimage
from app.core.surface_options import SurfaceAnalysisError, SurfaceOptions

AUTO_TYPES: Final = {'FFT Auto':'FFT Fixed','Gauss Spline Auto':'Gauss Spline',
                    'Robust Gauss Spline Auto':'Robust Gauss Spline'}
FILTER_TRIM_LAYERS: Final = 5


def resolve_auto(options: SurfaceOptions, shape: tuple[int, int]) -> SurfaceOptions:
    """按原矩阵短边解析实际物理截止值，保留 Auto 类型以便追溯。"""
    if options.filter_type not in AUTO_TYPES or options.filter_mode == 'Off':
        return options
    size = min(shape)
    high = 30/size
    low = 10/size
    if options.filter_type == 'FFT Auto':
        fft_size = 1 << (max(shape)-1).bit_length()
        low = int(low*fft_size+.5)/fft_size
        high = int(high*fft_size+.5)/fft_size
    used = high if options.filter_mode != 'High Pass' else low
    if used > .5 or (options.filter_type == 'FFT Auto' and used == .5):
        raise SurfaceAnalysisError('矩阵尺寸不足以使用原厂 Auto 截止值，请增大数据区域或选择手动截止。')
    return replace(options, low_frequency=low*1000/options.pixel_size_um,
                   high_frequency=high*1000/options.pixel_size_um)


def trim_filter_edges(data: np.ndarray) -> np.ndarray:
    """逐层重新识别外部背景；孔洞在与外部连通后参与下一层裁剪。"""
    valid = np.isfinite(data)
    for _ in range(FILTER_TRIM_LAYERS):
        valid &= ndimage.binary_erosion(ndimage.binary_fill_holes(valid))
    return np.where(valid,data,np.nan)
