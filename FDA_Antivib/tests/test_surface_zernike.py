import numpy as np
import pytest
from pathlib import Path


def test_fringe_terms_use_explicit_unnormalized_numbering():
    from app.core.surface_zernike import fringe_basis
    x = np.array([0., .3, -.4]); y = np.array([0., .2, .1])
    basis = fringe_basis(x, y, 9)
    expected = [np.ones(3), x, y, 2*(x*x+y*y)-1, x*x-y*y, 2*x*y,
                (3*(x*x+y*y)-2)*x, (3*(x*x+y*y)-2)*y,
                6*(x*x+y*y)**2-6*(x*x+y*y)+1]
    np.testing.assert_allclose(np.stack(expected, axis=-1), basis, atol=1e-14)


def test_zernike_selective_removal_preserves_unselected_defocus_and_masks():
    from app.core.surface_zernike import fit_zernike
    y, x = np.indices((31, 31), dtype=float)
    xx, yy = (x-15)/14, (y-15)/14
    data = 40+3*xx-2*yy+7*(2*(xx*xx+yy*yy)-1)
    data[14, 14] = np.nan
    result = fit_zernike(data, 15., 15., 14., 9, (1, 2, 3))
    np.testing.assert_allclose([40, 3, -2, 7, 0, 0, 0, 0, 0], result.coefficients, atol=1e-12)
    expected = 7*(2*(xx*xx+yy*yy)-1)
    expected[(xx*xx+yy*yy > 1) | ~np.isfinite(data)] = np.nan
    np.testing.assert_allclose(expected, result.processed, atol=1e-11)
    assert np.isnan(result.processed[0, 0])


def test_zernike_rejects_invalid_aperture_and_rank():
    from app.core.surface_zernike import fit_zernike
    with pytest.raises(ValueError, match='半径'):
        fit_zernike(np.ones((9, 9)), 4, 4, 0, 9, (1,))
    data = np.full((9, 9), np.nan); data[4] = 1
    with pytest.raises(ValueError, match='退化|不足'):
        fit_zernike(data, 4, 4, 4, 9, (1,))


@pytest.mark.parametrize('index', range(10))
def test_fringe_coefficients_and_selected_removal_match_native(index):
    from app.core.surface_zernike import fit_zernike
    terms = (index//2+2)**2
    selected = (1, 2, 3) if index % 2 == 0 else tuple(range(1, terms+1))
    with np.load(Path(__file__).parent/'data'/'surface_zernike_native.npz') as native:
        result = fit_zernike(native[f'input_{index}'], 18, 16, 14, terms, selected)
        np.testing.assert_allclose(native[f'coefficients_{index}'], result.coefficients, atol=1e-10)
        np.testing.assert_allclose(native[f'expected_{index}'], result.processed, atol=1e-10)


def test_zernike_page_uses_explicit_aperture_and_displays_coefficients(qtbot, tmp_path):
    from app.gui.surface_analysis_page import PlaneAnalysisPage
    data = np.ones((25, 25))*10
    source = tmp_path/'circle.txt'; np.savetxt(source, data)
    page = PlaneAnalysisPage(); qtbot.addWidget(page)
    controls = page.surface_controls
    controls.remove.setCurrentIndex(controls.remove.findData('Zernike Fringe'))
    controls.spikes.setChecked(False)
    controls.zernike.center_x.setValue(12)
    controls.zernike.center_y.setValue(12)
    controls.zernike.radius.setValue(10)
    page.file_edit.setText(str(source)); page._run_analysis()
    qtbot.waitUntil(lambda: page._result is not None and page._analysis_thread is None, timeout=8000)
    assert 10 == pytest.approx(page._result.coefficients[0])
    assert np.isnan(page._result.processed[0, 0])
    assert '拟合参数' in page.metrics.toPlainText()
