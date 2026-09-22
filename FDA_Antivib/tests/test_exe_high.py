import numpy as np
from app.reconstruction.high_residual import subtract_plane
from app.reconstruction.high_residual import subtract_histogram_mode,wrap_high_residual


def test_high_plane_residual_preserves_holes_and_uses_truncation():
    y,x=np.mgrid[:5,:7]
    plane=.7*x-1.1*y+4.2
    values=np.trunc(plane)+23
    values[2,3]=np.nan
    result=subtract_plane(values,np.array([.7,-1.1,4.2]))
    np.testing.assert_allclose(np.full(34,23),result[np.isfinite(values)])
    assert np.isnan(result[2,3])


def test_high_residual_wrap_keeps_053_period_band():
    raw=np.array([[-54.,-53.,-51.,51.,53.,54.,254.]])
    # float32(0.53)*100略小于53，原厂cvttss2si向零截断为52。
    np.testing.assert_array_equal([[46,47,-51,51,-47,-46,-46]],wrap_high_residual(raw,100))


def test_histogram_mode_uses_peak_bin_instead_of_mean():
    raw=np.array([[0.,20.,20.,20.,20.,98.,np.nan]])
    result,offset=subtract_histogram_mode(raw)
    assert offset==20
    np.testing.assert_allclose(raw[:,:-1]-20,result[:,:-1])
