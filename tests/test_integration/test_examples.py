"""Smoke tests: run each example script and confirm it does not crash."""

import runpy
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
    "get_coordinates.py",
]

EXAMPLES_3D = [
    "heatmap_3d.py",
    "region_annotation_custom.py",
    "plan.py",
]


@pytest.mark.parametrize(
    "example",
    EXAMPLES_2D,
    ids=EXAMPLES_2D,
)
def test_example_2d(example):
    """Run a 2D example and confirm it does not crash."""
    script = EXAMPLES_DIR / example
    assert script.exists(), f"Example not found: {script}"

    plt.close("all")
    try:
        with patch.object(plt, "show"):
            runpy.run_path(str(script))
    finally:
        plt.close("all")


# @pytest.mark.skipif(
#     sys.platform != "linux" or not os.environ.get("DISPLAY"),
#     reason="3D tests need xvfb (Linux only)",
# )
@pytest.mark.parametrize(
    "example",
    EXAMPLES_3D,
    ids=EXAMPLES_3D,
)
def test_example_3d(example):
    """Run a 3D example and confirm it does not crash."""
    script = EXAMPLES_DIR / example
    assert script.exists(), f"Example not found: {script}"

    runpy.run_path(str(script))
