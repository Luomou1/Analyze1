"""原厂 FftFilter 默认预处理、补边及 Gaussian 分支的独立实现。"""
import numpy as np

from app.core.surface_operators import remove_form
from app.core.surface_options import SurfaceAnalysisError, SurfaceOptions
from app.core.surface_cutoff import sinusoid_transfer


def fft_filter(data: np.ndarray, options: SurfaceOptions) -> np.ndarray:
    """按物理频率滤波；返回原有效域上的高度，单位不变。"""
    residual, fitted, _ = remove_form(data, 'Cylinder')
    valid = np.isfinite(data)
    # 默认 Padding=3：各轴补到同一个二次幂，奇数差值的余量留在末端。
    size = 1 << (max(data.shape)-1).bit_length()
    offsets = [(size-length)//2 for length in data.shape]
    region = tuple(slice(start, start+length) for start, length in zip(offsets, data.shape))
    padded = np.zeros((size, size))
    padded[region] = np.where(valid, residual, 0)
    spectrum = np.fft.rfft2(padded)
    squared_frequency = np.fft.fftfreq(size)[:, None]**2 + np.fft.rfftfreq(size)[None, :]**2

    def low_pass(frequency_per_mm: float) -> np.ndarray:
        frequency_per_pixel = frequency_per_mm * options.pixel_size_um / 1000
        if options.cutoff_shape == 'Sinusoid':
            # EXE 0x9EF8F0 以最近频格检查截止，避开 DLL 未初始化的 Nyquist 边界。
            cutoff_bin = int(frequency_per_pixel*size+.5)
            if not 1 <= cutoff_bin < size//2:
                raise SurfaceAnalysisError('Sinusoid 截止需位于第 1 个频格至 Nyquist 前一格，请调整截止或选择 Gaussian。')
        # 截止处振幅为 1/2；不是硬掩码，也不是以功率 1/2 定义的 -3 dB。
        transfer = (sinusoid_transfer(size, frequency_per_pixel)[:, :size//2+1]
                    if options.cutoff_shape == 'Sinusoid' else
                    np.exp2(-squared_frequency / frequency_per_pixel**2))
        return np.fft.irfft2(spectrum*transfer, s=padded.shape)[region] + fitted

    # 既有参数接口为 str；未知值显式报业务错误，不能用断言代替输入校验。
    match options.filter_mode:  # noqa: MATCH_OK
        case 'Low Pass':
            result = low_pass(options.high_frequency)
        case 'High Pass':
            result = data - low_pass(options.low_frequency)
        case 'Band Pass':
            result = low_pass(options.high_frequency) - low_pass(options.low_frequency)
        case 'Band Reject':
            result = data - low_pass(options.high_frequency) + low_pass(options.low_frequency)
        case _:
            raise SurfaceAnalysisError('FFT 需要低通、高通、带通或带阻模式。')
    return np.where(valid, result, np.nan)
