"""四阶去形状：原厂残差、退化域和后台参数快照。"""
from pathlib import Path

import numpy as np
import pytest

from app.core.surface_operators import remove_form
from app.core.surface_options import SurfaceAnalysisError, SurfaceOptions
from app.core.surface_processing import process_surface
from app.gui.surface_analysis_page import PlaneAnalysisPage

REFERENCE = Path(__file__).parent / 'data' / 'surface_poly4_native.npz'


@pytest.mark.parametrize('index', range(4))
def test_fourth_order_matches_native_with_rectangles_and_holes(index: int) -> None:
    with np.load(REFERENCE) as reference:
        data = reference[f'input_{index}']
        result = process_surface(data, SurfaceOptions(remove='4th Order', remove_spikes=False))
        np.testing.assert_allclose(reference[f'expected_{index}_0'], result.processed, atol=1e-10, rtol=0)
        np.testing.assert_equal(data, result.original)
        assert result.coefficients.size == 15
        assert result.stats['valid_count'] == np.isfinite(data).sum()


def test_fourth_order_does_not_remove_total_degree_five_cross_term() -> None:
    y, x = np.mgrid[-1:1:11j, -1:1:13j]
    data = x**3*y**2
    residual, _, _ = remove_form(data, '4th Order')
    assert np.max(np.abs(residual)) > .01


@pytest.mark.parametrize('shape', [(3, 9), (9, 3), (2, 2)])
def test_fourth_order_rejects_rank_deficient_domain(shape: tuple[int, int]) -> None:
    with pytest.raises(SurfaceAnalysisError, match='退化'):
        remove_form(np.ones(shape), '4th Order')


def test_fourth_order_runs_through_page_worker(qtbot, tmp_path: Path) -> None:
    with np.load(REFERENCE) as reference:
        data = reference['input_0']
        expected = reference['expected_0_0']
    path = tmp_path / 'fourth_order.txt'
    np.savetxt(path, data)
    page = PlaneAnalysisPage()
    qtbot.addWidget(page)
    controls = page.surface_controls
    index = controls.remove.findData('4th Order')
    assert controls.remove.model().item(index).isEnabled()
    controls.remove.setCurrentIndex(index)
    controls.spikes.setChecked(False)
    page.file_edit.setText(str(path))
    page._run_analysis()
    qtbot.waitUntil(lambda: page._result is not None and page._analysis_thread is None, timeout=8000)
    np.testing.assert_allclose(expected, page._result.processed, atol=1e-10, rtol=0)

