import numpy as np
import pytest
from app.reconstruction.engine import analyze


def surface_cube(n=169):
    y,x=np.mgrid[:16,:20];h=n*.05/2+.003*x+.002*y+.0001*x*y
    z=np.arange(n)*.05
    cube=1500+800*np.exp(-.5*((z-h[...,None])/.25)**2)*np.cos(22*(z-h[...,None]))
    return cube,h


@pytest.mark.parametrize('mode',['normal','high','high2g'])
@pytest.mark.parametrize('n',[103,128,169,201])
def test_three_reconstruction_modes_have_physical_height(mode,n):
    cube,h=surface_cube(n);r=analyze(cube,.05,mode)
    assert r['analysis_method']==mode
    assert r['fft_length']==n
    np.testing.assert_allclose(h*1000,r['h'],atol=.5)
    assert np.isfinite(r['h_prime']).all()
    np.testing.assert_allclose(h*1000,r['h_prime'],atol=.08,rtol=0)
    if mode=='normal':np.testing.assert_array_equal(r['h'],r['h_prime'])


def test_unsupported_old_mode_is_rejected():
    with pytest.raises(ValueError,match='模式'):
        analyze(np.ones((2,2,100)),.05,'FDA')
