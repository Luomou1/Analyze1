from dataclasses import replace

import numpy as np
import pytest

from app.reconstruction.engine import analyze
from app.reconstruction.high1g import reconstruct_high1g
from app.reconstruction.spectrum import reconstruct_spectrum


def sample():
    y,x=np.mgrid[:8,:10];z=np.arange(169)*.05
    height=4+.005*x+.003*y
    cube=1500+800*np.exp(-.5*((z-height[...,None])/.3)**2)*np.cos(22*(z-height[...,None]))
    return cube,height


@pytest.mark.parametrize('unwrap',['exe','unwrap1'])
def test_high1g_physical_height_and_manual_carrier(unwrap):
    cube,height=sample()
    result=analyze(cube,.05,'high1g',unwrap_method=unwrap)
    np.testing.assert_allclose(height*1000,result['h_prime'],atol=.1,rtol=0)
    manual=analyze(cube,.05,'high1g',common_k0=10.7,unwrap_method=unwrap)
    assert manual['k0_value']==10.7
    assert manual['postprocessing']=='none'
    assert 'connected_mask' not in manual
    assert np.nanmax(abs(manual['h_prime']-manual['h']))<=manual['height_period_nm']/2+1e-8


def test_order_correction_preserves_half_period_and_does_not_connect_pixels():
    cube,_=sample();front=reconstruct_spectrum(cube,.05)
    # 周期=1 μm，故意让相邻像素跨越多个周期；每点应独立校正。
    delta=np.resize(np.array([.5,-.5,2.7,-2.7,.1,-.1]),front.valid.shape)
    zero=np.zeros_like(delta)
    front=replace(front,coarse_um=zero,reference_um=zero,slope=zero,
                  intercept=delta*2*np.pi)
    output,diagnostic=reconstruct_high1g(front,common_k0=np.pi)
    expected=np.resize(np.array([.5,-.5,-.3,.3,.1,-.1]),delta.shape)
    np.testing.assert_allclose(expected,output,atol=1e-14)
    # 挖去任意一个像素不能改变它的邻点。
    valid=front.valid.copy();valid[2,3]=False
    masked,_=reconstruct_high1g(replace(front,valid=valid),common_k0=np.pi)
    assert np.isnan(masked[2,3])
    np.testing.assert_array_equal(output[valid],masked[valid])
