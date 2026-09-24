"""显示保真与交互回归；不以截图像素代替高度数据验证。"""
import numpy as np
import pytest
import pyvista as pv

from app.plotting.preview import build_normalized_surface_grid


def test_background_surface_keeps_event_loop_alive_and_delivers_latest(qtbot):
    from threading import Event
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QWidget
    from app.plotting.surface_jobs import SurfaceJobs, prepare_surface
    owner = QWidget()
    qtbot.addWidget(owner)
    errors, results, ticks = [], [], []
    jobs = SurfaceJobs(owner, errors.append)
    release = Event()
    started = Event()
    def slow():
        started.set()
        assert release.wait(10)
        return "old"
    timer = QTimer(owner)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)
    jobs.submit(slow, results.append)
    try:
        qtbot.waitUntil(lambda: started.is_set() and len(ticks) >= 3)
        jobs.submit(lambda: prepare_surface(np.arange(256 * 256.).reshape(256, 256)), results.append)
        qtbot.waitUntil(lambda: len(results) == 1, timeout=15000)
        assert results[0].mesh.n_points == 256 * 256
        assert "Normals" in results[0].mesh.point_data
    finally:
        release.set()
    qtbot.waitUntil(lambda: not jobs._signals, timeout=15000)
    assert len(results) == 1
    assert not errors
    jobs.submit(lambda: "closed", results.append)
    jobs.cancel()
    qtbot.waitUntil(lambda: not jobs._signals)
    assert len(results) == 1


@pytest.mark.parametrize("analysis", [False, True])
def test_large_surface_finishes_in_background(qtbot, analysis):
    from app.plotting.preview import SurfaceCanvas
    from app.gui.surface_analysis_page import Surface3DCanvas
    canvas = Surface3DCanvas() if analysis else SurfaceCanvas()
    qtbot.addWidget(canvas)
    data = np.arange(256 * 256.).reshape(256, 256)
    try:
        canvas.draw_surface(data, "Async")
        assert canvas._surface_jobs.callback is not None
        qtbot.waitUntil(lambda: canvas._surface_jobs.callback is None, timeout=15000)
        plotter = canvas._plotter if analysis else canvas.plotter
        meshes = [a.mapper.dataset for a in plotter.actors.values()
                  if isinstance(a, pv.Actor) and isinstance(a.mapper, pv.DataSetMapper)
                  and "height" in a.mapper.dataset.point_data]
        assert len(meshes) == 1
        assert meshes[0].n_points == data.size
        np.testing.assert_array_equal(np.sort(meshes[0]["height"]), data.ravel())
    finally:
        canvas.shutdown()


@pytest.mark.parametrize("kind", ["preview", "plane", "step"])
def test_height_lighting_has_no_control_and_preserves_data(qtbot, kind):
    from app.plotting.preview import HeatmapCanvas
    from app.gui.surface_analysis_page import ResultFigureCanvas, StepSelectionCanvas
    from matplotlib import colormaps
    if kind == "preview":
        canvas = HeatmapCanvas(surface_style=True)
        draw = lambda data: canvas.draw_map(data, "Height")
    elif kind == "plane":
        canvas = ResultFigureCanvas()
        draw = lambda data: canvas.draw_height_map(data, "Height")
    else:
        canvas = StepSelectionCanvas()
        draw = canvas.start_point_selection
    qtbot.addWidget(canvas)
    data = np.arange(20.).reshape(4, 5)
    original = data.copy()
    draw(data)
    image = canvas.figure.axes[0].images[0]
    limits = image.get_clim()
    assert not hasattr(canvas, "brightness")
    assert len(canvas.figure.axes[0].images) == 2
    shading = np.asarray(canvas.figure.axes[0].images[1].get_array())
    assert shading.shape == (*data.shape, 4)
    assert np.any(shading[..., 3] > 0)
    assert limits == image.get_clim()
    np.testing.assert_array_equal(original, data)
    draw(data)
    assert len(canvas.figure.axes[0].images) == 2
    assert len(canvas.figure.axes) == (1 if kind == "step" else 2)



@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("analysis", [False, True])
def test_2d_3d_share_colors_and_compact_upper_right_bar(qtbot, flat, analysis):
    from app.plotting.preview import HeatmapCanvas, SurfaceCanvas
    from app.gui.surface_analysis_page import ResultFigureCanvas, Surface3DCanvas
    data = np.full((5, 7), 12.0) if flat else np.arange(35.0).reshape(5, 7) ** 2
    image_canvas = ResultFigureCanvas() if analysis else HeatmapCanvas(surface_style=True)
    surface_canvas = Surface3DCanvas() if analysis else SurfaceCanvas()
    qtbot.addWidget(image_canvas)
    qtbot.addWidget(surface_canvas)
    try:
        if analysis:
            image_canvas.draw_height_map(data, "Height")
        else:
            image_canvas.draw_map(data, "Height")
        surface_canvas.draw_surface(data, "Height")
        image = image_canvas.figure.axes[0].images[0]
        plotter = surface_canvas._plotter if analysis else surface_canvas.plotter
        actor = next(a for a in plotter.actors.values()
                     if isinstance(a, pv.Actor)
                     and hasattr(a.mapper, "dataset") and "height" in a.mapper.dataset.point_data)
        assert image.get_clim() == pytest.approx(actor.mapper.scalar_range)
        expected = image.cmap(np.linspace(0, 1, 256), bytes=True)
        expected = actor.mapper.lookup_table.values
        np.testing.assert_allclose(image.cmap(np.linspace(0, 1, 256), bytes=True), expected, atol=1)
        bar = next(iter(plotter.scalar_bars.values()))
        assert bar.GetPosition()[0] >= .8
        assert bar.GetPosition()[1] >= .65
        assert bar.GetHeight() <= .25
        assert bar.GetWidth() <= .13
        assert bar.GetNumberOfLabels() == 2
        assert bar.GetBarRatio() <= .15
    finally:
        surface_canvas.shutdown()


def test_native_grid_preserves_single_pixel_feature_and_last_row():
    data = np.zeros((301, 407))
    data[151, 203] = 17.25
    grid, values, x, y, bounds = build_normalized_surface_grid(data)
    assert data.shape == values.shape
    assert data.size == grid.n_points
    assert (0, 406) == (x[0], x[-1])
    assert (0, 300) == (y[0], y[-1])
    assert (0, 17.25) == bounds
    np.testing.assert_array_equal(data, values)


def test_small_relief_on_large_offset_is_not_lost_to_float32():
    data = 1e9 + np.array([[0, .01], [.02, .03]])
    grid, values, *_ = build_normalized_surface_grid(data)
    assert 4 == np.unique(grid.points[:, 2]).size
    np.testing.assert_array_equal(data, values)


def test_missing_point_does_not_create_bridging_surface_cells():
    data = np.ones((5, 5))
    data[2, 2] = np.nan
    from app.plotting.surface_style import surface_geometry
    grid, values, *_ = build_normalized_surface_grid(data)
    surface = surface_geometry(grid, values)
    assert 24 == surface.n_cells
    assert np.isfinite(surface.points).all()
    assert np.isfinite(surface["height"]).all()


def test_display_controls_reuse_mesh_and_preserve_heights(qtbot):
    from app.plotting.preview import SurfaceCanvas

    canvas = SurfaceCanvas()
    qtbot.addWidget(canvas)
    data = np.arange(120, dtype=float).reshape(10, 12)
    original = data.copy()
    try:
        canvas.draw_surface(data, "Surface")
        actors = [a for a in canvas.plotter.actors.values() if hasattr(a, "mapper") and a.mapper is not None]
        actor = next(a for a in actors if hasattr(a.mapper, "dataset") and a.mapper.dataset.n_points >= data.size)
        mesh = actor.mapper.dataset
        points = mesh.points.copy()
        canvas.surface_controls.relief.setValue(60)
        canvas.surface_controls.lighting.setChecked(False)
        assert mesh is actor.mapper.dataset
        np.testing.assert_array_equal(points, mesh.points)
        np.testing.assert_array_equal(original, data)
        assert .6 == pytest.approx(canvas.plotter.scale[2])
        assert (0, 1, 0, 9 / 11, 0, .6) == pytest.approx(canvas.plotter.renderer.cube_axes_actor.bounds)
        assert (0, 119) == pytest.approx(canvas.plotter.renderer.cube_axes_actor.z_axis_range)
        assert not actor.prop.lighting
    finally:
        canvas.shutdown()


def test_diagnostic_heatmap_keeps_original_style(qtbot):
    from app.plotting.preview import HeatmapCanvas
    diagnostic = HeatmapCanvas()
    height = HeatmapCanvas(surface_style=True)
    qtbot.addWidget(diagnostic)
    qtbot.addWidget(height)
    data = np.arange(12).reshape(3, 4)
    diagnostic.draw_map(data, "Phase")
    height.draw_map(data, "Height")
    assert "viridis" == diagnostic.axes.images[0].get_cmap().name
    assert "jet" == height.axes.images[0].get_cmap().name
    assert "nearest" == height.axes.images[0].get_interpolation()


@pytest.mark.parametrize("kind", ["preview", "analysis", "comparison"])
def test_fully_missing_surface_clears_without_rendering_a_false_plane(qtbot, kind):
    from app.plotting.preview import SurfaceCanvas, ComparisonCanvas
    from app.gui.surface_analysis_page import Surface3DCanvas
    classes = {"preview": SurfaceCanvas, "analysis": Surface3DCanvas, "comparison": ComparisonCanvas}
    canvas = classes[kind]()
    qtbot.addWidget(canvas)
    data = np.full((5, 7), np.nan)
    try:
        if kind == "comparison":
            canvas.draw_comparison(data, data, "Missing")
        else:
            canvas.draw_surface(data, "Missing")
        plotter = canvas._plotter if kind == "analysis" else canvas.plotter
        if plotter is not None:
            import pyvista as pv
            assert not any(isinstance(a, pv.Actor) and isinstance(a.mapper, pv.DataSetMapper)
                           and "height" in a.mapper.dataset.point_data and a.mapper.dataset.n_cells
                           for a in plotter.actors.values())
    finally:
        canvas.shutdown()
