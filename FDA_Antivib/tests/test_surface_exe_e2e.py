"""真实 MetroPro 进程、Micro.app 控件和 DAT 导出的整链固定对照。"""
import json
from pathlib import Path

import numpy as np
import pytest

from app.core.surface_options import SurfaceOptions
from app.core.surface_processing import process_surface
from app.core.surface_operators import trim_edges

FIXTURE = Path(__file__).parent/'data'/'surface_exe_e2e_native.npz'
METADATA = json.loads(FIXTURE.with_suffix('.npz.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('case', METADATA['cases'], ids=lambda case: case['name'])
def test_micro_app_export_matches_independent_pipeline(case):
    # 原厂导出含整数相位量化；父子窗口在同一原始数据上依次处理。
    with np.load(FIXTURE, allow_pickle=False) as native:
        data = native['input_'+case['name']].copy()
        if 'preprocess' in case:
            pre = case['preprocess']
            if pre['kind'] == 'circle':
                y, x = np.indices(data.shape)
                pixel = pre['pixel_size_m']
                mask = ((x-pre['center_x_m']/pixel)**2 + (y-pre['center_y_m']/pixel)**2
                        <= (pre['radius_m']/pixel)**2)
                data[~mask] = np.nan
            else:
                data = trim_edges(data, pre['count'], pre['mode'])
        parent = process_surface(data, SurfaceOptions(**case['parent']))
        result = process_surface(parent.processed, SurfaceOptions(**case['options']))
        np.testing.assert_array_equal(np.isfinite(native[case['name']]), np.isfinite(result.processed))
        np.testing.assert_allclose(native[case['name']], result.processed, atol=METADATA['atol_nm'], rtol=0)


def test_default_fill_uses_exe_branch():
    assert 'MetroPro EXE' == SurfaceOptions().data_fill_method
