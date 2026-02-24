"""Unit tests for AtlasS3Heatmap._prepare_colors method."""

import numpy as np
import pytest


class MockAtlasS3Heatmap:
    """Minimal mock to test _prepare_colors in isolation."""

    def __init__(self, values: dict):
        self.values = values

    def _prepare_colors(
        self,
        cmap: str,
        vmin: float | None,
        vmax: float | None,
    ) -> tuple[dict, float, float]:
        """Copy of the method from AtlasS3Heatmap for isolated testing."""
        from brainrender import settings
        from vedo.colors import color_map as vedo_map_color

        # Calculate vmin/vmax from data if not provided
        not_nan = [v for v in self.values.values() if not np.isnan(v)]
        if len(not_nan) == 0:
            _vmax, _vmin = np.nan, np.nan
        else:
            _vmax, _vmin = max(not_nan), min(not_nan)

        if _vmax == _vmin:
            _vmin = _vmax * 0.5

        final_vmin: float = _vmin if vmin is None else vmin
        final_vmax: float = _vmax if vmax is None else vmax

        colors: dict = {
            r: vedo_map_color(v, name=cmap, vmin=final_vmin, vmax=final_vmax)
            for r, v in self.values.items()
        }
        colors["root"] = settings.ROOT_COLOR

        return colors, final_vmin, final_vmax


class TestVminVmaxCalculation:
    """Tests for automatic vmin/vmax calculation."""

    def test_calculates_vmin_vmax_from_values(self):
        """Should calculate vmin/vmax from the range of values."""
        mock = MockAtlasS3Heatmap(
            values={"VIS": 10.0, "MOp": 50.0, "SSp": 30.0}
        )

        _, vmin, vmax = mock._prepare_colors("viridis", None, None)

        assert vmin == 10.0
        assert vmax == 50.0

    def test_handles_negative_values(self):
        """Should correctly calculate range with negative values."""
        mock = MockAtlasS3Heatmap(values={"VIS": -20.0, "MOp": 30.0})

        _, vmin, vmax = mock._prepare_colors("viridis", None, None)

        assert vmin == -20.0
        assert vmax == 30.0


class TestAllNanValues:
    """Tests for handling all-NaN input values."""

    def test_all_nan_returns_nan_vmin_vmax(self):
        """All NaN values should result in NaN vmin and vmax."""
        mock = MockAtlasS3Heatmap(values={"VIS": np.nan, "MOp": np.nan})

        _, vmin, vmax = mock._prepare_colors("viridis", None, None)

        assert np.isnan(vmin)
        assert np.isnan(vmax)

    def test_all_nan_still_creates_colors_dict(self):
        """All NaN values should still produce a colors dictionary."""
        mock = MockAtlasS3Heatmap(values={"VIS": np.nan, "MOp": np.nan})

        colors, _, _ = mock._prepare_colors("viridis", None, None)

        assert "VIS" in colors
        assert "MOp" in colors
        assert "root" in colors


class TestMixedNanValues:
    """Tests for handling mixed NaN and valid values."""

    def test_nan_values_excluded_from_range_calc(self):
        """NaN values should be excluded when calculating vmin/vmax."""
        mock = MockAtlasS3Heatmap(
            values={"VIS": 10.0, "MOp": np.nan, "SSp": 50.0, "ACA": np.nan}
        )

        _, vmin, vmax = mock._prepare_colors("viridis", None, None)

        assert vmin == 10.0
        assert vmax == 50.0


class TestSingleValue:
    """Tests for single value / all identical values edge case."""

    def test_single_value_adjusts_vmin(self):
        """Single value should adjust vmin to half of vmax."""
        mock = MockAtlasS3Heatmap(values={"VIS": 100.0})

        _, vmin, vmax = mock._prepare_colors("viridis", None, None)

        assert vmax == 100.0
        assert vmin == 50.0  # 100 * 0.5

    def test_all_identical_values_adjusts_vmin(self):
        """All identical values should adjust vmin to half of vmax."""
        mock = MockAtlasS3Heatmap(
            values={"VIS": 20.0, "MOp": 20.0, "SSp": 20.0}
        )

        _, vmin, vmax = mock._prepare_colors("viridis", None, None)

        assert vmax == 20.0
        assert vmin == 10.0  # 20 * 0.5

    def test_single_zero_value(self):
        """Single zero value: vmax=0, vmin=0*0.5=0 (edge case)."""
        mock = MockAtlasS3Heatmap(values={"VIS": 0.0})

        _, vmin, vmax = mock._prepare_colors("viridis", None, None)

        assert vmax == 0.0
        assert vmin == 0.0  # 0 * 0.5 = 0

    def test_single_negative_value(self):
        """Single negative value: vmin should be half (less negative)."""
        mock = MockAtlasS3Heatmap(values={"VIS": -100.0})

        _, vmin, vmax = mock._prepare_colors("viridis", None, None)

        assert vmax == -100.0
        assert vmin == -50.0  # -100 * 0.5


class TestUserOverride:
    """Tests for user-provided vmin/vmax override."""

    def test_user_vmin_overrides_calculated(self):
        """User-provided vmin should override calculated value."""
        mock = MockAtlasS3Heatmap(values={"VIS": 10.0, "MOp": 50.0})

        _, vmin, vmax = mock._prepare_colors("viridis", vmin=0.0, vmax=None)

        assert vmin == 0.0
        assert vmax == 50.0

    def test_user_vmax_overrides_calculated(self):
        """User-provided vmax should override calculated value."""
        mock = MockAtlasS3Heatmap(values={"VIS": 10.0, "MOp": 50.0})

        _, vmin, vmax = mock._prepare_colors("viridis", vmin=None, vmax=100.0)

        assert vmin == 10.0
        assert vmax == 100.0

    def test_both_user_overrides(self):
        """Both user-provided vmin and vmax should override calculated values."""
        mock = MockAtlasS3Heatmap(values={"VIS": 10.0, "MOp": 50.0})

        _, vmin, vmax = mock._prepare_colors("viridis", vmin=-10.0, vmax=100.0)

        assert vmin == -10.0
        assert vmax == 100.0

    def test_inverted_user_range(self):
        """User can provide inverted range (vmin > vmax) for reversed colormap."""
        mock = MockAtlasS3Heatmap(values={"VIS": 10.0, "MOp": 50.0})

        _, vmin, vmax = mock._prepare_colors("viridis", vmin=100.0, vmax=0.0)

        assert vmin == 100.0
        assert vmax == 0.0


class TestColorsDict:
    """Tests for the colors dictionary output."""

    def test_colors_dict_contains_all_regions(self):
        """Colors dict should contain all input regions."""
        mock = MockAtlasS3Heatmap(
            values={"VIS": 10.0, "MOp": 50.0, "SSp": 30.0}
        )

        colors, _, _ = mock._prepare_colors("viridis", None, None)

        assert "VIS" in colors
        assert "MOp" in colors
        assert "SSp" in colors

    def test_colors_dict_contains_root(self):
        """Colors dict should always contain 'root' key."""
        mock = MockAtlasS3Heatmap(values={"VIS": 10.0})

        colors, _, _ = mock._prepare_colors("viridis", None, None)

        assert "root" in colors

    def test_color_values_are_rgb_compatible(self):
        """Color values should be numpy arrays or lists with 3 elements."""
        mock = MockAtlasS3Heatmap(values={"VIS": 10.0, "MOp": 50.0})

        colors, _, _ = mock._prepare_colors("viridis", None, None)

        for region in ["VIS", "MOp"]:
            color = colors[region]
            # Should have 3 components (RGB)
            assert len(color) == 3
            # Values should be in [0, 1] range
            assert all(0 <= c <= 1 for c in color)


class TestColormapUsage:
    """Tests for colormap parameter handling."""

    @pytest.mark.parametrize(
        "cmap",
        [
            pytest.param("viridis", id="viridis"),
            pytest.param("plasma", id="plasma"),
            pytest.param("Reds", id="Reds"),
            pytest.param("Blues", id="Blues"),
        ],
    )
    def test_different_colormaps_work(self, cmap):
        """Various colormap names should be accepted."""
        mock = MockAtlasS3Heatmap(values={"VIS": 10.0, "MOp": 50.0})

        colors, vmin, vmax = mock._prepare_colors(cmap, None, None)

        assert "VIS" in colors
        assert "MOp" in colors


class TestEdgeCases:
    """Tests for various edge cases."""

    def test_empty_values_dict(self):
        """Empty values dict should work but only have 'root'."""
        mock = MockAtlasS3Heatmap(values={})

        colors, vmin, vmax = mock._prepare_colors("viridis", None, None)

        assert "root" in colors
        assert len(colors) == 1
        assert np.isnan(vmin)
        assert np.isnan(vmax)

    def test_very_large_values(self):
        """Very large values should be handled."""
        mock = MockAtlasS3Heatmap(values={"VIS": 1e10, "MOp": 1e12})

        _, vmin, vmax = mock._prepare_colors("viridis", None, None)

        assert vmin == 1e10
        assert vmax == 1e12

    def test_very_small_values(self):
        """Very small values should be handled."""
        mock = MockAtlasS3Heatmap(values={"VIS": 1e-10, "MOp": 1e-8})

        _, vmin, vmax = mock._prepare_colors("viridis", None, None)

        assert vmin == 1e-10
        assert vmax == 1e-8

    def test_wide_magnitude_range(self):
        """Wide range of magnitudes should be handled."""
        mock = MockAtlasS3Heatmap(values={"VIS": 0.001, "MOp": 1000.0})

        _, vmin, vmax = mock._prepare_colors("viridis", None, None)

        assert vmin == 0.001
        assert vmax == 1000.0
