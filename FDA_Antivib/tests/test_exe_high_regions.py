from pathlib import Path
import numpy as np
import pytest
from app.reconstruction.high_regions import reject_discontinuities,unwrap_regions

@pytest.mark.parametrize('case',range(6))
def test_high_helpers_match_actual_exe(case):
    with np.load(Path(__file__).parent/'data'/f'exe_high_regions_{case}.npz') as r:
        filtered=reject_discontinuities(r['input'])
        np.testing.assert_allclose(r['filtered'],filtered,equal_nan=True)
        output,_,_=unwrap_regions(filtered)
        np.testing.assert_allclose(r['unwrapped'],output,equal_nan=True)
