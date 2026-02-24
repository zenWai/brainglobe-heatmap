"""Examples as pixel-comparison and smoke tests."""

import os
import runpy
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import pytest
from brainrender import settings

matplotlib.use("Agg")
settings.INTERACTIVE = False
settings.OFFSCREEN = True

EXAMPLES_DIR = Path(__file__).parents[2] / "examples"
OUTPUT_DIR = Path(__file__).parents[2] / "test_example_outputs"
BASELINE_DIR = str(Path(__file__).parent / "baseline")

NOT_TESTED = [
    "heatmap_human_brain.py",  # non-default atlas
    "heatmap_spinal_cord.py",  # non-default atlas
    "heatmap_zebrafish.py",  # non-default atlas
]

EXAMPLES_2D = [
    "heatmap_2d.py",
    "heatmap_2d_subplots.py",
    "slicer_2D.py",
    "region_annotation.py",
    "region_annotation_specified.py",
    "cellfinder_cell_density.py",
]

EXAMPLES_3D = [
    "heatmap_3d.py",
    "region_annotation_custom.py",
    "plan.py",
    "heatmap_3d_new_example.py",
]

EXAMPLES_NONVISUAL = [
    "get_coordinates.py",
]

SCENE_VARIABLE_NAME = {
    "heatmap_3d.py": "scene",
    "region_annotation_custom.py": "f",
    "plan.py": "planner",
    "heatmap_3d_new_example.py": "scene",
}


@pytest.fixture(scope="session")
def output_dir():
    """Create a fresh output directory for 3D screenshots."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()
    yield OUTPUT_DIR


@pytest.mark.mpl_image_compare(
    baseline_dir=BASELINE_DIR,
    tolerance=2,
    savefig_kwarg={"dpi": 150, "bbox_inches": "tight"},
    style="default",
)
@pytest.mark.parametrize(
    "example",
    EXAMPLES_2D,
    ids=EXAMPLES_2D,
)
def test_example_2d(example):
    """2D examples,
    return its figure for pytest-mpl comparison.
    """
    script = EXAMPLES_DIR / example
    assert script.exists(), f"Example not found: {script}"

    plt.close("all")
    try:
        with patch.object(plt, "show"):
            runpy.run_path(str(script))

        fig = plt.gcf()
        assert fig.get_axes(), f"{example} produced no axes"
        return fig
    finally:
        if plt.get_fignums():
            plt.close("all")


@pytest.mark.mpl_image_compare(
    baseline_dir=BASELINE_DIR,
    tolerance=28,
    savefig_kwarg={"dpi": 150, "bbox_inches": "tight"},
    style="default",
)
@pytest.mark.skipif(
    sys.platform != "linux" or not os.environ.get("DISPLAY"),
    reason="3D tests need xvfb (Linux only)",
)
@pytest.mark.parametrize(
    "example",
    EXAMPLES_3D,
    ids=EXAMPLES_3D,
)
def test_example_3d(example, output_dir):
    """3D examples,
    screenshot and return as a figure for pytest-mpl comparison.
    """
    script = EXAMPLES_DIR / example
    assert script.exists(), f"Example not found: {script}"

    plt.close("all")
    try:
        with patch.object(plt, "show"):
            namespace = runpy.run_path(str(script))

        var = SCENE_VARIABLE_NAME[example]
        obj = namespace.get(var)
        assert obj is not None, f"{example}: expected variable '{var}'"
        # Resolve to Scene (e.g. plan objects wrap their scene)
        scene = getattr(obj, "scene", obj)

        stem = Path(example).stem
        filepath = str(output_dir / f"{stem}.png")
        scene.screenshot(filepath)
        scene.close()

        # Wrap screenshot in a figure for pytest-mpl comparison
        plt.close("all")
        img = plt.imread(filepath)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis("off")
        return fig
    finally:
        if plt.get_fignums():
            plt.close("all")


@pytest.mark.parametrize(
    "example",
    EXAMPLES_NONVISUAL,
    ids=EXAMPLES_NONVISUAL,
)
def test_example_nonvisual(example):
    """Confirms it completes without error."""
    script = EXAMPLES_DIR / example
    assert script.exists(), f"Example not found: {script}"

    runpy.run_path(str(script))
