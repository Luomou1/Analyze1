import numpy as np
import pytest
from app.reconstruction.engine import preview_spectrum,analyze

@pytest.mark.parametrize("unwrap",["exe","unwrap1"])
@pytest.mark.parametrize("radius",[1,3,5])
def test_preview_common_k_matches_reconstruction(unwrap,radius):
    z=np.arange(169)*.05
    y,x=np.mgrid[:8,:10]
    height=4+.003*x+.002*y
    carrier=21.5+.04*x
    cube=1500+800*np.exp(-.5*((z-height[...,None])/.3)**2)*np.cos(carrier[...,None]*(z-height[...,None]))
    cube[0,0]=1500
    result=analyze(cube,.05,unwrap_method=unwrap,window_size=radius)
    preview=preview_spectrum(cube,.05,unwrap_method=unwrap,window_size=radius)
    assert preview['candidate_count']==79
    assert preview['k0_value']==result['k0_value']
    assert abs(preview['k0_value']-preview['k_axis'][np.argmax(preview['spectrum'])]) > 1e-4

def test_no_valid_pixels_reports_error():
    with pytest.raises(ValueError,match="有效像素"):
        preview_spectrum(np.ones((2,2,100)),.05)


def test_manual_common_k_controls_high2g_height():
    y,x=np.mgrid[:16,:20];z=np.arange(169)*.05
    h=4+.003*x+.002*y+.0001*x*y
    cube=1500+800*np.exp(-.5*((z-h[...,None])/.3)**2)*np.cos(22*(z-h[...,None]))
    automatic=analyze(cube,.05,mode='high2g')
    same=analyze(cube,.05,mode='high2g',common_k0=automatic['k0_value'])
    manual=analyze(cube,.05,mode='high2g',common_k0=10.7)
    assert manual['k0_value']==10.7
    assert manual['k0_source']=='manual'
    np.testing.assert_allclose(automatic['h_prime'],same['h_prime'],atol=1e-8)
    np.testing.assert_array_equal(automatic['h'],manual['h'])
    assert np.nanmax(abs(automatic['h_prime']-manual['h_prime']))>1e-3

@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf')])
def test_bad_manual_common_k_rejected(value):
    with pytest.raises(ValueError,match='公共K0'):
        analyze(np.ones((3,3,20)),.05,mode='high2g',common_k0=value)
