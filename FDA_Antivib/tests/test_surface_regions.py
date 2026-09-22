"""Min Area Size 原厂有效区域和处理链对照。"""
from pathlib import Path

import numpy as np
import pytest

from app.core.surface_options import SurfaceAnalysisError, SurfaceOptions
from app.core.surface_processing import process_surface
from app.gui.surface_controls import SurfaceControls

REFERENCE = Path(__file__).parent / 'data' / 'surface_regions_native.npz'


@pytest.mark.parametrize('minimum', [1, 3, 4, 5, 6, 7, 9])
def test_min_area_matches_native_four_connected_inclusive_threshold(minimum: int) -> None:
    with np.load(REFERENCE) as reference:
        data = reference['input']
        expected = np.where(reference[f'labels_{minimum}'] != 2147483640, data, np.nan)
        result = process_surface(data, SurfaceOptions(remove='None', remove_spikes=False, min_area=minimum))
        np.testing.assert_equal(expected, result.processed)
        np.testing.assert_equal(data, result.original)


def test_min_area_rejects_empty_result() -> None:
    with np.load(REFERENCE) as reference:
        with pytest.raises(SurfaceAnalysisError, match='有效'):
            process_surface(reference['input'], SurfaceOptions(remove='None', remove_spikes=False, min_area=10))


def test_min_area_control_enters_options_snapshot(qtbot) -> None:
    controls = SurfaceControls()
    qtbot.addWidget(controls)
    with qtbot.waitSignal(controls.changed):
        controls.min_area.setValue(7)
    assert controls.options().min_area == 7
