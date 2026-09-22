import numpy as np

from app.reconstruction.diagnostics import build_pixel_analysis
from app.gui.pixel_window import PixelAnalysisWindow


def test_pixel_window_updates_valid_invalid_valid(qtbot):
    z = np.arange(169)*.05
    signal = 1900+1000*np.exp(-.5*((z-4)/.3)**2)*np.cos(22*(z-4))
    cube = np.stack([signal, np.full_like(signal,1900), np.roll(signal,8)])[None]
    window = PixelAnalysisWindow()
    qtbot.addWidget(window)
    for x in [0,1,2]:
        payload = build_pixel_analysis(cube,x,0,.05,global_k0_value=11,maximum_scan_value=4095)
        window.update_analysis(payload,'h_prime','weighted','metro')
        window.canvas.draw()
        assert f'x={x}, y=0' in window.header_label.text()
        np.testing.assert_allclose(cube[0,x],window.raw_axes.lines[0].get_ydata())


def test_diagnostic_spectrum_and_fit_only_show_supported_data():
    z = np.arange(169)*.05
    signal = 1900+1000*np.exp(-.5*((z-4)/.3)**2)*np.cos(22*(z-4))
    p = build_pixel_analysis(signal[None,None],0,0,.05,global_k0_value=11,maximum_scan_value=4095)
    assert abs(p['amplitude_y'][0]) < 1e-9
    assert np.isfinite(p['fit_mask_phase_y']).all()
    assert p['fit_k_x'][0] == p['fit_mask_k_x'][0]
    assert p['fit_k_x'][-1] == p['fit_mask_k_x'][-1]


def test_full_phase_display_does_not_expand_fitting_band():
    z=np.arange(169)*.05
    signal=1900+1000*np.exp(-.5*((z-4)/.3)**2)*np.cos(22*(z-4))
    payload=build_pixel_analysis(signal[None,None],0,0,.05)
    assert len(payload['k_x']) == 85
    assert np.isfinite(payload['phase_raw_y'][1:]).all()
    assert np.isfinite(payload['phase_original_y'][1:]).all()
    assert payload['fit_point_count'] == len(payload['fit_k_x'])
    assert payload['fit_point_count'] < len(payload['k_x'])
    np.testing.assert_allclose(np.diff(payload['fit_phase_y'],n=2),0,atol=1e-12)


def test_coordinate_controls_refresh_signal_and_clamp_bounds(qtbot):
    z=np.arange(169)*.05
    signal=1900+1000*np.exp(-.5*((z-4)/.3)**2)*np.cos(22*(z-4))
    cube=np.stack([signal,np.roll(signal,8)])[None]
    window=PixelAnalysisWindow();qtbot.addWidget(window)
    window.set_image_shape(1,2)
    events=[]
    def refresh(layer,x,y):
        events.append((x,y))
        window.update_analysis(build_pixel_analysis(cube,x,y,.05),layer,'weighted','exe')
    window.coordinates_changed.connect(refresh)
    refresh('h_prime',0,0)
    window.x_coordinate.setValue(1)
    assert events == [(0,0),(1,0)]
    np.testing.assert_array_equal(cube[0,1],window.raw_axes.lines[0].get_ydata())
    window.y_coordinate.setValue(100)
    assert window.y_coordinate.value() == 0
    assert window.x_coordinate.maximum() == 1
