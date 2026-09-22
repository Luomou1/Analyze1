"""高级球面/锥面与原厂几何拟合残差对照，应用高度单位为 nm。"""
from pathlib import Path
import numpy as np
import pytest
from app.core.surface_options import SurfaceOptions
from app.core.surface_processing import process_surface
from app.gui.surface_analysis_page import PlaneAnalysisPage

DATA=Path(__file__).parent/'data'


@pytest.mark.parametrize('index',range(4))
def test_fixed_sphere_matches_native_convex_and_concave(index: int) -> None:
    with np.load(DATA/'surface_sphere_native.npz') as r:
        result=process_surface(r[f'input_{index}']*1000,SurfaceOptions(remove='Fixed Rad Sphere',
            remove_spikes=False,pixel_size_um=1,sphere_radius_mm=float(r[f'radius_{index}'])/1000))
        np.testing.assert_allclose(r[f'expected_{index}_0']*1000,result.processed,atol=2e-5,rtol=0)


@pytest.mark.parametrize('index',range(4))
@pytest.mark.parametrize('fixed',[True,False])
def test_cone_matches_native_tilt_and_sign(index: int,fixed: bool) -> None:
    with np.load(DATA/'surface_cone_native.npz') as r:
        options=SurfaceOptions(remove='Fixed Angle Cone' if fixed else 'Variable Angle Cone',
            remove_spikes=False,pixel_size_um=1,cone_angle_deg=float(r[f'angle_{index}']))
        result=process_surface(r[f'input_{index}']*1000,options)
        np.testing.assert_allclose(r[f'expected_{index}_{0 if fixed else 1}']*1000,
                                   result.processed,atol=.002,rtol=0)


@pytest.mark.parametrize(('mode','fixture','key'),[
    ('Fixed Rad Sphere','surface_sphere_native.npz','expected_0_0'),
    ('Fixed Angle Cone','surface_cone_native.npz','expected_0_0'),
    ('Variable Angle Cone','surface_cone_native.npz','expected_0_1')])
def test_physical_form_runs_through_page_worker(qtbot,tmp_path: Path,mode: str,fixture: str,key: str) -> None:
    with np.load(DATA/fixture) as r:
        data=r['input_0']*1000;expected=r[key]*1000
    path=tmp_path/'form.txt';np.savetxt(path,data)
    page=PlaneAnalysisPage();qtbot.addWidget(page)
    controls=page.surface_controls
    index=controls.remove.findData(mode)
    assert controls.remove.model().item(index).isEnabled()
    controls.remove.setCurrentIndex(index)
    assert controls.pixel_size.isEnabled()
    controls.pixel_size.setValue(1);controls.radius.setValue(.05)
    controls.cone_angle.setValue(120);controls.spikes.setChecked(False)
    page.file_edit.setText(str(path));page._run_analysis()
    qtbot.waitUntil(lambda:page._result is not None and page._analysis_thread is None,timeout=10000)
    np.testing.assert_allclose(expected,page._result.processed,atol=.002,rtol=0)
    assert page._result.options.remove==mode


@pytest.mark.parametrize('changes',[
    {'remove':'Fixed Rad Sphere','sphere_radius_mm':0},
    {'remove':'Fixed Rad Sphere','sphere_radius_mm':float('nan')},
    {'remove':'Fixed Angle Cone','cone_angle_deg':0},
    {'remove':'Fixed Angle Cone','cone_angle_deg':3},
    {'remove':'Variable Angle Cone','cone_angle_deg':181},
])
def test_invalid_physical_parameters_rejected(changes: dict) -> None:
    with pytest.raises(ValueError):
        SurfaceOptions(pixel_size_um=1,**changes)


def test_variable_cone_can_estimate_angle_without_input() -> None:
    with np.load(DATA/'surface_cone_native.npz') as r:
        result=process_surface(r['input_0']*1000,SurfaceOptions(remove='Variable Angle Cone',
            remove_spikes=False,pixel_size_um=1))
        np.testing.assert_allclose(r['expected_0_1']*1000,result.processed,atol=.002,rtol=0)
