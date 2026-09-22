import json
from pathlib import Path
import numpy as np
import pytest

FIXTURE = Path(__file__).parent/'data'/'surface_differential_native.npz'
CASES = json.loads(FIXTURE.with_suffix('.npz.json').read_text(encoding='utf-8'))['cases']


@pytest.mark.parametrize('case', CASES, ids=[case['key'] for case in CASES])
def test_differential_filler_matches_native(case):
    from app.core.surface_differential import differential_fill
    with np.load(FIXTURE) as native:
        data = native[case['key']+'_input']
        expected = native[case['key']+'_expected']
        actual, count = differential_fill(data, case['maximum'], case['mode'])
    np.testing.assert_allclose(expected, actual, atol=1e-9, rtol=0)
    assert count == np.isfinite(expected).sum()-np.isfinite(data).sum()
