"""Auto 参数、EXE 频率滤波裁边和样条带阻。"""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from app.core.surface_options import SurfaceOptions
from app.core.surface_processing import process_surface
from app.core.surface_operators import trim_edges
from app.core.surface_filter_controls import trim_filter_edges
from app.core.surface_filter_controls import resolve_auto
from app.gui.surface_analysis_page import PlaneAnalysisPage


@pytest.mark.parametrize('index',range(4))
def test_filter_trim_matches_exe_outside_five_layers(index: int) -> None:
    path=Path(__file__).parent/'data'/'surface_filter_control_native.npz'
    with np.load(path) as r:
        np.testing.assert_equal(r[f'expected_{index}'],trim_filter_edges(r[f'input_{index}']))


@pytest.mark.parametrize('size',[64,96,100,128])
def test_fft_auto_quantization_matches_exe(size: int) -> None:
    options=SurfaceOptions(filter_type='FFT Auto',filter_mode='Band Pass',pixel_size_um=1000)
    result=resolve_auto(options,(size,size))
    with np.load(Path(__file__).parent/'data'/'surface_filter_control_native.npz') as r:
        np.testing.assert_equal(r[f'cutoffs_{size}'],[result.low_frequency,result.high_frequency])


@pytest.mark.parametrize(('auto','fixed'),[
    ('FFT Auto','FFT Fixed'),('Gauss Spline Auto','Gauss Spline'),
    ('Robust Gauss Spline Auto','Robust Gauss Spline')])
def test_auto_uses_shorter_dimension_and_preserves_resolved_snapshot(auto: str,fixed: str) -> None:
    y,x=np.indices((96,128),dtype=float)
    data=np.cos(x*.37)+np.sin(y*.71)
    options=SurfaceOptions(remove='None',remove_spikes=False,pixel_size_um=2,
                           filter_type=auto,filter_mode='Band Pass')
    actual=process_surface(data,options)
    low = 13/128 if auto == 'FFT Auto' else 10/96
    high = 40/128 if auto == 'FFT Auto' else 30/96
    expected=process_surface(data,replace(options,filter_type=fixed,
        low_frequency=low*500,high_frequency=high*500))
    np.testing.assert_allclose(expected.processed,actual.processed,atol=1e-11)
    assert actual.options.filter_type == auto
    assert actual.options.low_frequency == pytest.approx(low*500)
    assert actual.options.high_frequency == pytest.approx(high*500)


@pytest.mark.parametrize(('algorithm','fixture'),[
    ('Gauss Spline','surface_spline_native.npz'),
    ('Robust Gauss Spline','surface_robust_spline_native.npz')])
def test_spline_band_reject_matches_native(algorithm: str,fixture: str) -> None:
    with np.load(Path(__file__).parent/'data'/fixture) as r:
        result=process_surface(r['input_3'],SurfaceOptions(remove='None',remove_spikes=False,
            filter_type=algorithm,filter_mode='Band Reject',pixel_size_um=10,
            low_frequency=10,high_frequency=20))
        np.testing.assert_allclose(r['expected_3_3'],result.processed,atol=2e-11)


def test_fft_trim_is_applied_after_filtering() -> None:
    data=np.random.default_rng(5).normal(size=(31,37));data[15,18]=np.nan
    options=SurfaceOptions(remove='None',remove_spikes=False,filter_mode='Low Pass',
        filter_type='FFT Fixed',pixel_size_um=10,high_frequency=20)
    untrimmed=process_surface(data,options)
    result=process_surface(data,replace(options,filter_trim=True))
    np.testing.assert_allclose(trim_filter_edges(untrimmed.processed),result.processed)


@pytest.mark.parametrize('algorithm',['FFT Auto','Gauss Spline Auto','Robust Gauss Spline Auto'])
def test_auto_page_displays_resolved_cutoffs_without_invalidating_result(qtbot,tmp_path: Path,algorithm: str) -> None:
    y,x=np.indices((96,128),dtype=float);data=np.cos(x*.37)+np.sin(y*.71)
    path=tmp_path/'auto.txt';np.savetxt(path,data)
    page=PlaneAnalysisPage();qtbot.addWidget(page)
    controls=page.surface_controls
    controls.remove.setCurrentIndex(0);controls.spikes.setChecked(False)
    controls.filter_mode.setCurrentText('Band Pass');controls.filter_type.setCurrentText(algorithm)
    controls.pixel_size.setValue(2)
    assert not controls.low_frequency.isEnabled()
    page.file_edit.setText(str(path));page._run_analysis()
    qtbot.waitUntil(lambda:page._result is not None and page._analysis_thread is None,timeout=10000)
    assert controls.low_frequency.value()==pytest.approx(page._result.options.low_frequency)
    assert controls.high_frequency.value()==pytest.approx(page._result.options.high_frequency)


def test_sinusoid_selection_invalidates_previous_result(qtbot) -> None:
    page=PlaneAnalysisPage();qtbot.addWidget(page)
    controls=page.surface_controls
    controls.filter_mode.setCurrentText('Low Pass');controls.filter_type.setCurrentText('FFT Fixed')
    with qtbot.waitSignal(controls.changed):
        controls.cutoff_shape.setCurrentText('Sinusoid')


def test_disabled_auto_does_not_require_calibration_or_display_reciprocal(qtbot) -> None:
    page=PlaneAnalysisPage();qtbot.addWidget(page)
    options=SurfaceOptions(filter_type='FFT Auto',filter_mode='Off')
    page.surface_controls.show_resolved_options(options)
    result=process_surface(np.arange(100,dtype=float).reshape(10,10),replace(options,remove_spikes=False))
    assert result.options.low_frequency==0
