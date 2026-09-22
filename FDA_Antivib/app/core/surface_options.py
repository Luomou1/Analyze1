"""两种表面分析共用的不可变处理参数。"""
from dataclasses import dataclass
import math


class SurfaceAnalysisError(ValueError):
    """输入或处理参数不能产生可用的测量结果。"""


@dataclass(frozen=True, slots=True)
class SurfaceOptions:
    remove: str = 'Plane'
    trim: int = 0
    trim_mode: str = 'All'
    min_area: int = 0
    remove_spikes: bool = True
    spike_height: float = 2.5
    data_fill: bool = False
    data_fill_max: int = 25
    data_fill_mode: int = 0
    data_fill_method: str = 'MetroPro EXE'
    filter_mode: str = 'Off'
    filter_type: str = 'Average'
    window_size: int = 3
    filter_trim: bool = False
    low_frequency: float = 0.0
    high_frequency: float = 0.0
    pixel_size_um: float = 0.48
    sphere_radius_mm: float = 0.0
    cutoff_shape: str = 'Gaussian'
    cone_angle_deg: float = 0.0
    zernike_center_x: float = 0.0
    zernike_center_y: float = 0.0
    zernike_radius: float = 0.0
    zernike_terms: int = 9
    zernike_remove_terms: tuple[int, ...] = (1, 2, 3)

    def __post_init__(self) -> None:
        physical = self.remove in ('Fixed Rad Sphere','Fixed Angle Cone','Variable Angle Cone')
        if self.remove not in ('None', 'Piston', 'Plane', 'Sphere', 'Cylinder', 'Three Points', '4th Order', 'Zernike Fringe') and not physical:
            raise SurfaceAnalysisError('所选形状去除尚未完成原厂对照。')
        if self.remove == 'Zernike Fringe':
            if not all(math.isfinite(v) for v in (self.zernike_center_x, self.zernike_center_y, self.zernike_radius)) or self.zernike_radius <= 0:
                raise SurfaceAnalysisError('请填写泽尼克孔径中心和正的半径（像素）。')
            if self.zernike_terms not in (4, 9, 16, 25, 36):
                raise SurfaceAnalysisError('Fringe 拟合项数须为 4、9、16、25 或 36。')
            if len(set(self.zernike_remove_terms)) != len(self.zernike_remove_terms) or any(j < 1 or j > self.zernike_terms for j in self.zernike_remove_terms):
                raise SurfaceAnalysisError('泽尼克扣除项须唯一且位于拟合范围内。')
        if physical:
            if not math.isfinite(self.pixel_size_um) or self.pixel_size_um <= 0:
                raise SurfaceAnalysisError('球面/锥面拟合需要正的横向标定（μm/pixel）。')
            if self.remove == 'Fixed Rad Sphere':
                if not math.isfinite(self.sphere_radius_mm) or self.sphere_radius_mm == 0:
                    raise SurfaceAnalysisError('固定球面半径必须为非零有限值（mm）。')
            elif not math.isfinite(self.cone_angle_deg) or not (
                5 <= abs(self.cone_angle_deg) <= 180 or
                (self.remove == 'Variable Angle Cone' and self.cone_angle_deg == 0)):
                raise SurfaceAnalysisError('全锥角需为 ±5～±180°；自由锥角可填 0 自动估计。')
        if self.trim_mode not in ('All', 'Outside') or not 0 <= self.trim <= 10:
            raise SurfaceAnalysisError('Trim 必须为 0～10，模式为 All 或 Outside。')
        if not isinstance(self.min_area, int) or not 0 <= self.min_area <= 9999999:
            raise SurfaceAnalysisError('Min Area Size 必须为 0～9999999 的整数。')
        if self.data_fill_max < 0:
            raise SurfaceAnalysisError('区域大小不能为负数。')
        if self.data_fill_mode not in (0, 1, 2):
            raise SurfaceAnalysisError('补洞模式必须为限面积内部孔、全部内部孔或含边界孔。')
        if self.data_fill_method not in ('MetroPro EXE', 'Polynomial', 'Differential'):
            raise SurfaceAnalysisError('未知补洞算法。')
        if self.data_fill and self.data_fill_method == 'MetroPro EXE' and self.data_fill_mode != 0:
            raise SurfaceAnalysisError('MetroPro EXE 补洞只支持原厂范围；其他范围请选择 Polynomial 或 Differential。')
        if self.window_size < 3 or self.window_size > 99 or self.window_size % 2 == 0:
            raise SurfaceAnalysisError('滤波窗口必须为 3～99 的奇数。')
        if not math.isfinite(self.spike_height) or self.spike_height <= 0:
            raise SurfaceAnalysisError('尖峰阈值倍数必须为正的有限数。')
        if self.filter_mode not in ('Off', 'Low Pass', 'High Pass', 'Band Pass', 'Band Reject'):
            raise SurfaceAnalysisError('未知滤波模式。')
        if self.filter_mode == 'Off':
            return
        if self.filter_type in ('FFT Fixed', 'Gauss Spline', 'Robust Gauss Spline',
                                'FFT Auto', 'Gauss Spline Auto', 'Robust Gauss Spline Auto'):
            if not math.isfinite(self.pixel_size_um) or self.pixel_size_um <= 0:
                raise SurfaceAnalysisError('频率滤波需要正的横向标定（μm/pixel），请填写实际像素尺寸。')
            if self.cutoff_shape not in ('Gaussian', 'Sinusoid'):
                raise SurfaceAnalysisError('截止形状必须为 Gaussian/Sinusoid。')
            if self.filter_type.endswith(' Auto'):
                return
            frequencies = []
            if self.filter_mode != 'High Pass':
                frequencies.append(self.high_frequency)
            if self.filter_mode != 'Low Pass':
                frequencies.append(self.low_frequency)
            if any(not math.isfinite(value) or value <= 0 for value in frequencies):
                raise SurfaceAnalysisError('所用截止频率必须为正的有限数（1/mm）。')
            scaled = [value*self.pixel_size_um/1000 for value in frequencies]
            if any(not math.isfinite(value*value) or value*value == 0 for value in scaled):
                raise SurfaceAnalysisError('频率与横向标定的乘积超出可计算范围。')
            if self.filter_mode in ('Band Pass', 'Band Reject') and self.low_frequency >= self.high_frequency:
                raise SurfaceAnalysisError('带通/带阻需要 Low Freq 小于 High Freq。')
            if self.filter_type in ('Gauss Spline', 'Robust Gauss Spline'):
                if any(value*self.pixel_size_um/1000 > .5 for value in frequencies):
                    raise SurfaceAnalysisError('样条截止频率不能超过横向采样的 Nyquist 频率。')
        elif self.filter_mode in ('Band Pass', 'Band Reject'):
            raise SurfaceAnalysisError('窗口滤波不支持带通/带阻，请选择已验证的频率滤波。')
