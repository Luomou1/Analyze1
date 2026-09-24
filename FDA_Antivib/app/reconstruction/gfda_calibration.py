"""GFDA 校正表：配置绑定的 JSON，以及有明确采样约束的原厂 AVC 交换。"""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from app.reconstruction.gfda_phase import GfdaError

FORMAT_VERSION: Final = 1
CARRIER_RELATIVE_TOLERANCE: Final = .01


@dataclass(frozen=True, slots=True)
class Calibration:
    coefficients: NDArray[np.float64]
    step_um: float
    carrier: float
    phase_half_width: float
    source: str

    @property
    def radius(self) -> int:
        return len(self.coefficients)//2

    def validate(self) -> None:
        values = self.coefficients
        if (values.ndim != 1 or len(values) < 3 or len(values) % 2 != 1
                or not np.isfinite(values).all() or np.any(values <= 0)
                or not np.isclose(values[self.radius], 1., rtol=0, atol=1e-12)
                or not np.isfinite([self.step_um, self.carrier, self.phase_half_width]).all()
                or min(self.step_um, self.carrier, self.phase_half_width) <= 0):
            raise GfdaError("校正表必须为有限正数、奇数长度，中心系数为 1。")

    def check_configuration(self, step_um: float, carrier: float, phase_half_width: float) -> None:
        self.validate()
        if (not np.isclose(self.step_um, step_um, rtol=1e-6, atol=0)
                or not np.isclose(self.carrier, carrier, rtol=CARRIER_RELATIVE_TOLERANCE, atol=0)
                or not np.isclose(self.phase_half_width, phase_half_width, rtol=1e-6, atol=0)):
            raise GfdaError("校正文件的名义步长、载频或选点相位宽度与本次扫描不匹配。")


def save_calibration(path: str | Path, calibration: Calibration) -> None:
    """JSON 保存完整尺度信息；AVC 只允许原厂四分之一条纹、21 项约定。"""
    calibration.validate()
    target = Path(path)
    if target.suffix.lower() == '.avc':
        if (calibration.radius != 10
                or not np.isclose(2*calibration.carrier*calibration.step_um, np.pi/2, rtol=CARRIER_RELATIVE_TOLERANCE)
                or not np.isclose(calibration.phase_half_width, np.pi)):
            raise GfdaError("AVC 仅支持 21 项和名义四分之一条纹采样，请保存为 .gfda.json。")
        content = ''.join(f'{value:.20e}\t{i}\t{i-10}\n'
                          for i, value in enumerate(calibration.coefficients))
    else:
        content = json.dumps({'format': 'gfda-calibration', 'version': FORMAT_VERSION,
                              'coefficients': calibration.coefficients.tolist(),
                              'step_um': calibration.step_um, 'carrier': calibration.carrier,
                              'phase_half_width': calibration.phase_half_width,
                              'source': calibration.source}, ensure_ascii=False, indent=2)
    target.write_text(content, encoding='utf-8')


def load_calibration(path: str | Path, step_um: float, carrier: float,
                     phase_half_width: float | None = None) -> Calibration:
    """拒绝缺项、非法系数和尺度不匹配；不对缺失项补 1。"""
    target = Path(path)
    try:
        content = target.read_text(encoding='utf-8-sig')
        if target.suffix.lower() == '.avc':
            values = np.array([float(line.split()[0]) for line in content.splitlines() if line.strip()])
            if len(values) != 21 or not np.isclose(2*carrier*step_um, np.pi/2, rtol=CARRIER_RELATIVE_TOLERANCE):
                raise GfdaError("原厂 AVC 缺少尺度元数据，仅接受 21 项且名义四分之一条纹的配置。")
            calibration = Calibration(values, step_um, carrier, np.pi, str(target))
        else:
            data = json.loads(content)
            if data['format'] != 'gfda-calibration' or data['version'] != FORMAT_VERSION:
                raise GfdaError("不支持的 GFDA 校正文件格式或版本。")
            calibration = Calibration(np.asarray(data['coefficients'], dtype=float),
                                      float(data['step_um']), float(data['carrier']),
                                      float(data['phase_half_width']), str(target))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        raise GfdaError(f"无法读取校正文件：{error}") from error
    calibration.check_configuration(step_um, carrier,
                                    calibration.phase_half_width if phase_half_width is None else phase_half_width)
    if not np.pi <= calibration.phase_half_width <= 2*np.pi:
        raise GfdaError("校正文件的选点相位半宽超出支持范围 [π,2π]。")
    return calibration
