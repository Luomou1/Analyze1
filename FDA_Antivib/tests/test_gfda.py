"""GFDA 的物理尺度、运动恢复与失败域；信号由独立的运动真值生成。"""
import numpy as np
import pytest

from app.reconstruction.engine import analyze
from app.reconstruction.gfda_phase import estimate_pair, GfdaError
from app.reconstruction.gfda_spectrum import nonuniform_spectrum, reconstruct_nonuniform


def moving_surface(n=128, step=.05, amplitude=.012):
    y, x = np.mgrid[:12, :48]
    height = 1.8 + .032*x + .007*y + .00004*x*y
    nominal = np.arange(n)*step
    # 改变采样步长时保持相同的物理运动波长，避免同时改变被测运动。
    motion = amplitude*np.sin(2*np.pi*nominal/.85)
    positions = nominal + motion
    offset = positions[None, None, :] - height[..., None]
    cube = 1500 + 850*np.exp(-.5*(offset/.28)**2)*np.cos(22*offset)
    return cube, height, positions


@pytest.mark.parametrize('phase_step,group,expected', [
    (np.pi/2,50,1.570835429574), (np.pi/2,46,1.564926400876),
    (np.pi/2,54,1.576446216745), (.8,50,.793373229388),
    (.8,46,.804385593663), (.8,54,.794371848720),
    (2.2,50,2.198249123886), (2.2,46,2.192735739649), (2.2,54,2.187412924156),
])
def test_phase_pair_matches_native_normal_path(phase_step, group, expected):
    peaks = np.linspace(10, 90, 1000)
    heights = np.rint(peaks*100)*.01
    frames = np.rint(128+90*np.exp(-.5*((np.array([50, 51])[:, None]-peaks)/8)**2)
                     *np.cos(phase_step*(np.array([50, 51])[:, None]-peaks)+.2))
    pair = estimate_pair(heights, np.full(1000, 128.), frames[0], frames[1],
                         group+.5, phase_step/2, phase_half_width=np.pi)
    # 前一轮独立调用原厂 0x140AB4880；此断言并非候选实现自生成的预期。
    assert pair.phase_rad == pytest.approx(expected, abs=1e-11)


def test_many_pixels_with_no_phase_diversity_are_rejected():
    with pytest.raises(GfdaError, match="相位|秩|参考"):
        estimate_pair(np.ones(500), np.full(500, 100.), np.full(500, 130.),
                      np.full(500, 110.), 1., 11.)


@pytest.mark.parametrize('n', [63, 96, 128])
def test_uniform_coordinates_reduce_to_same_window_dft(n):
    values = np.random.default_rng(n).normal(size=(3,n))
    z = np.arange(n)*.05
    bins = np.arange(1, n//2)
    actual = nonuniform_spectrum(values, z, bins, .05)
    np.testing.assert_allclose(np.conj(np.fft.rfft(values, axis=1))[:, bins], actual,
                               atol=2e-12, rtol=1e-12)


def test_known_nonuniform_coordinates_recover_height_and_reference():
    cube, height, positions = moving_surface(amplitude=.025)
    front = reconstruct_nonuniform(cube, positions, .05, maximum=4095)
    assert front.valid.mean() > .98
    np.testing.assert_allclose(height[front.valid], front.coarse_um[front.valid], atol=.002)
    translated = reconstruct_nonuniform(cube, positions+7., .05, maximum=4095)
    np.testing.assert_allclose(front.coarse_um+7., translated.coarse_um, atol=1e-10)


@pytest.mark.parametrize('n, step', [(128,.05), (201,.035)])
def test_gfda_recovers_motion_from_images_and_improves_surface(n, step):
    cube, height, positions = moving_surface(n, step)
    baseline = analyze(cube, step, mode='high2g')
    result = analyze(cube, step, mode='gfda')
    assert result['analysis_method'] == 'gfda'
    assert result['gfda_applied'] is True
    assert result['fft_length'] == n
    valid = result['valid_mask'] & baseline['valid_mask']
    assert valid.mean() > .8
    error = result['h_prime'][valid]/1000-height[valid]
    old_error = baseline['h_prime'][valid]/1000-height[valid]
    assert np.std(error) < np.std(old_error)*.8
    estimated = result['scan_positions_used_um']
    measured = result['scan_step_confidence'] > 0
    assert measured.sum() > 15
    assert np.sqrt(np.mean((np.diff(estimated)[measured]-np.diff(positions)[measured])**2)) < .004


def test_gfda_does_not_silently_fall_back_on_flat_phase_distribution():
    z = np.arange(128)*.05
    signal = 1500+850*np.exp(-.5*((z-3)/.28)**2)*np.cos(22*(z-3))
    with pytest.raises(GfdaError, match="相位|参考|高度"):
        analyze(np.broadcast_to(signal, (10,12,128)), .05, mode='gfda')


@pytest.mark.parametrize('n,step', [(128,.05),(201,.035)])
def test_no_vibration_keeps_relative_surface_within_half_nm(n, step):
    cube, height, _ = moving_surface(n, step, amplitude=0)
    result = analyze(cube, step, mode='gfda')
    error = result['h_prime']/1000-height
    # 只允许去活塞，保留倾斜/尺度误差；0.5nm 与既有粗高度回归预算一致。
    assert np.std(error) < .0005


@pytest.mark.parametrize('maximum', [255,4095])
def test_integer_intensity_gfda_improves_relative_height(maximum):
    cube, height, _ = moving_surface()
    cube = np.rint(cube*maximum/4095)
    normal = analyze(cube, .05, mode='high2g', maximum=maximum)
    gfda = analyze(cube, .05, mode='gfda', maximum=maximum)
    assert np.std(gfda['h_prime']/1000-height) < np.std(normal['h_prime']/1000-height)*.8


def test_reverse_pair_is_reported_instead_of_sorted_or_clipped():
    height = np.linspace(-.5,.5,200)
    first = 100+50*np.cos(22*height)
    second = 100+50*np.cos(22*height+.6)
    with pytest.raises(GfdaError, match='反向'):
        estimate_pair(height,np.full(200,100.),first,second,0.,11.)


def test_gfda_reports_invalid_origin():
    cube, _, _ = moving_surface()
    with pytest.raises(GfdaError, match='起点'):
        analyze(cube,.05,mode='gfda',start_um=np.nan)
