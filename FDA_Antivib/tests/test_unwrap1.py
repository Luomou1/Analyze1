import numpy as np
import pytest
from app.reconstruction.spectrum import phase_fit
from app.reconstruction.engine import analyze
from app.reconstruction.diagnostics import build_pixel_analysis

@pytest.mark.parametrize("radius",[1,3,5])
def test_user_band_and_restored_slope(radius):
    bins=np.arange(2,25)
    phase=-1.2*np.pi*bins
    power=100-(bins-12)**2*.1
    spectrum=np.sqrt(power)*np.exp(1j*phase)
    b,a,_,valid,left,right,u,_=phase_fit(spectrum[None],bins,0,"unwrap1",radius)
    assert valid[0]
    assert right[0]-left[0]+1 == 2*radius+1
    np.testing.assert_allclose(-1.2*np.pi,b[0],atol=1e-6)
    np.testing.assert_allclose(np.diff(u[0,np.isfinite(u[0])]),-1.2*np.pi,atol=1e-6)

@pytest.mark.parametrize("mode",["normal","high","high2g"])
def test_unwrap1_keeps_physical_height(mode):
    y,x=np.mgrid[:16,:20];height=4+.003*x+.002*y+.0001*x*y
    z=np.arange(169)*.05
    cube=1500+800*np.exp(-.5*((z-height[...,None])/.25)**2)*np.cos(22*(z-height[...,None]))
    result=analyze(cube,.05,mode,unwrap_method="unwrap1",window_size=3)
    np.testing.assert_allclose(height*1000,result['h_prime'],atol=.15,rtol=0)
    diagnostic=build_pixel_analysis(cube,0,0,.05,unwrap_method="unwrap1",window_size=3)
    assert diagnostic['fit_point_count']==7
    np.testing.assert_allclose(np.diff(diagnostic['fit_phase_y'],n=2),0,atol=1e-12)
