"""补洞模式与相邻孔洞使用离线原厂固定输出。"""
import json
from pathlib import Path

import numpy as np
import pytest

from app.core.surface_processing import _fill_holes

FIXTURE = Path(__file__).parent/'data'/'surface_fill_boundaries_native.npz'
CASES = json.loads(FIXTURE.with_suffix('.npz.json').read_text(encoding='utf-8'))['cases']


@pytest.mark.parametrize('case', CASES, ids=[case['key'] for case in CASES])
def test_fill_boundary_matches_native(case):
    with np.load(FIXTURE) as native:
        data = native[case['key']+'_input']
        expected = native[case['key']+'_expected']
        actual, count = _fill_holes(data, case['maximum'], case['mode'])
    assert case['status'] == 0
    np.testing.assert_allclose(expected, actual, atol=1e-9, rtol=0)
    assert np.isfinite(expected).sum()-np.isfinite(data).sum() == count
