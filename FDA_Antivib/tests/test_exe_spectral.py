from pathlib import Path

import numpy as np

from app.reconstruction.high import histogram_phase_reference
from app.reconstruction.spectrum import phase_fit

DATA = Path(__file__).parent / 'data'


def test_weighted_fit_matches_original_exe_three_point_kernel():
    with np.load(DATA / 'exe_spectral_three_points.npz') as original:
        bins = np.arange(2, 5)
        spectrum = np.sqrt(original['power']) * np.exp(1j * original['phase']) * (-1.)**bins
        slope, intercept, _, valid, _, _, phase, _ = phase_fit(spectrum, bins, 0)
        assert valid.all()
        np.testing.assert_allclose(original['unwrapped'], phase, atol=1e-6)
        np.testing.assert_allclose(original['slope'], slope, atol=5e-6)
        np.testing.assert_allclose(original['intercept'], intercept + slope * 2, atol=5e-6)


def test_high_phase_histogram_matches_original_exe():
    with np.load(DATA / 'exe_high_phase_reference.npz') as original:
        actual = [histogram_phase_reference(alpha) for alpha in original['alpha']]
        np.testing.assert_allclose(original['reference'], actual, rtol=0, atol=1e-6)
