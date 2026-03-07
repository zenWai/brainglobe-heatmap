"""Unit tests for AtlasS3Heatmap._apply_hemisphere_filter method."""

import numpy as np
import pytest


class MockAtlasS3Heatmap:
    """Minimal mock to test _apply_hemisphere_filter in isolation."""

    def __init__(self, hemisphere: str, orientation: str):
        self.hemisphere = hemisphere
        self.orientation = orientation

    def _apply_hemisphere_filter(self, slice_data: np.ndarray) -> np.ndarray:
        """Copy of the method from AtlasS3Heatmap for isolated testing."""
        if self.hemisphere == "both":
            return slice_data

        if self.orientation == "sagittal":
            return slice_data

        midpoint = slice_data.shape[1] // 2

        result = slice_data.copy()
        if self.hemisphere == "left":
            result[:, midpoint:] = 0
        elif self.hemisphere == "right":
            result[:, :midpoint] = 0

        return result


class TestHemisphereBoth:
    """Tests for hemisphere='both' mode."""

    def test_both_returns_unchanged_data(self):
        """hemisphere='both' should return the exact same data."""
        mock = MockAtlasS3Heatmap(hemisphere="both", orientation="frontal")
        slice_data = np.ones((10, 20), dtype=int) * 5

        result = mock._apply_hemisphere_filter(slice_data)

        np.testing.assert_array_equal(result, slice_data)

    def test_both_returns_same_object(self):
        """hemisphere='both' should return the same array object (no copy)."""
        mock = MockAtlasS3Heatmap(hemisphere="both", orientation="frontal")
        slice_data = np.ones((10, 20), dtype=int)

        result = mock._apply_hemisphere_filter(slice_data)

        assert result is slice_data


class TestHemisphereLeft:
    """Tests for hemisphere='left' mode."""

    def test_left_zeros_right_half(self):
        """hemisphere='left' should zero out the right half of the image."""
        mock = MockAtlasS3Heatmap(hemisphere="left", orientation="frontal")
        slice_data = np.ones((10, 20), dtype=int)

        result = mock._apply_hemisphere_filter(slice_data)

        # Left half (columns 0-9) should be preserved
        assert np.all(result[:, :10] == 1)
        # Right half (columns 10-19) should be zeroed
        assert np.all(result[:, 10:] == 0)

    def test_left_does_not_modify_original(self):
        """hemisphere='left' should not modify the original array."""
        mock = MockAtlasS3Heatmap(hemisphere="left", orientation="frontal")
        slice_data = np.ones((10, 20), dtype=int)
        original_copy = slice_data.copy()

        mock._apply_hemisphere_filter(slice_data)

        np.testing.assert_array_equal(slice_data, original_copy)


class TestHemisphereRight:
    """Tests for hemisphere='right' mode."""

    def test_right_zeros_left_half(self):
        """hemisphere='right' should zero out the left half of the image."""
        mock = MockAtlasS3Heatmap(hemisphere="right", orientation="frontal")
        slice_data = np.ones((10, 20), dtype=int)

        result = mock._apply_hemisphere_filter(slice_data)

        # Left half (columns 0-9) should be zeroed
        assert np.all(result[:, :10] == 0)
        # Right half (columns 10-19) should be preserved
        assert np.all(result[:, 10:] == 1)

    def test_right_does_not_modify_original(self):
        """hemisphere='right' should not modify the original array."""
        mock = MockAtlasS3Heatmap(hemisphere="right", orientation="frontal")
        slice_data = np.ones((10, 20), dtype=int)
        original_copy = slice_data.copy()

        mock._apply_hemisphere_filter(slice_data)

        np.testing.assert_array_equal(slice_data, original_copy)


class TestSagittalOrientation:
    """Tests for sagittal orientation special handling."""

    @pytest.mark.parametrize(
        "hemisphere",
        [
            pytest.param("left", id="left"),
            pytest.param("right", id="right"),
            pytest.param("both", id="both"),
        ],
    )
    def test_sagittal_bypasses_filtering(self, hemisphere):
        """Sagittal orientation should return data unchanged regardless of hemisphere."""
        mock = MockAtlasS3Heatmap(
            hemisphere=hemisphere, orientation="sagittal"
        )
        slice_data = np.ones((10, 20), dtype=int) * 7

        result = mock._apply_hemisphere_filter(slice_data)

        np.testing.assert_array_equal(result, slice_data)

    def test_sagittal_returns_same_object(self):
        """Sagittal orientation should return the same array object (no copy)."""
        mock = MockAtlasS3Heatmap(hemisphere="left", orientation="sagittal")
        slice_data = np.ones((10, 20), dtype=int)

        result = mock._apply_hemisphere_filter(slice_data)

        assert result is slice_data


class TestOtherOrientations:
    """Tests for frontal and horizontal orientations."""

    @pytest.mark.parametrize(
        "orientation",
        [
            pytest.param("frontal", id="frontal"),
            pytest.param("horizontal", id="horizontal"),
        ],
    )
    def test_left_hemisphere_filtering_works(self, orientation):
        """Left hemisphere filtering should work for frontal and horizontal."""
        mock = MockAtlasS3Heatmap(hemisphere="left", orientation=orientation)
        slice_data = np.ones((10, 20), dtype=int)

        result = mock._apply_hemisphere_filter(slice_data)

        midpoint = 10
        assert np.all(result[:, :midpoint] == 1)
        assert np.all(result[:, midpoint:] == 0)

    @pytest.mark.parametrize(
        "orientation",
        [
            pytest.param("frontal", id="frontal"),
            pytest.param("horizontal", id="horizontal"),
        ],
    )
    def test_right_hemisphere_filtering_works(self, orientation):
        """Right hemisphere filtering should work for frontal and horizontal."""
        mock = MockAtlasS3Heatmap(hemisphere="right", orientation=orientation)
        slice_data = np.ones((10, 20), dtype=int)

        result = mock._apply_hemisphere_filter(slice_data)

        midpoint = 10
        assert np.all(result[:, :midpoint] == 0)
        assert np.all(result[:, midpoint:] == 1)


class TestMidpointCalculation:
    """Tests for midpoint calculation edge cases."""

    def test_even_width_symmetric_split(self):
        """Even width should split exactly in half."""
        mock = MockAtlasS3Heatmap(hemisphere="left", orientation="frontal")
        # Width 100: midpoint = 50
        slice_data = np.ones((10, 100), dtype=int)

        result = mock._apply_hemisphere_filter(slice_data)

        assert np.all(result[:, :50] == 1)
        assert np.all(result[:, 50:] == 0)

    def test_odd_width_center_goes_to_right(self):
        """Odd width: center column goes to right hemisphere due to integer division."""
        mock = MockAtlasS3Heatmap(hemisphere="left", orientation="frontal")
        # Width 101: midpoint = 50, center column 50 goes to right half
        slice_data = np.arange(101).reshape(1, 101)

        result = mock._apply_hemisphere_filter(slice_data)

        # Left half preserved: columns 0-49
        assert np.all(result[:, :50] == slice_data[:, :50])
        # Right half zeroed: columns 50-100
        assert np.all(result[:, 50:] == 0)

    def test_width_2_minimum_split(self):
        """Width 2 should split into 1 column each."""
        mock = MockAtlasS3Heatmap(hemisphere="left", orientation="frontal")
        slice_data = np.array([[1, 2], [3, 4]])

        result = mock._apply_hemisphere_filter(slice_data)

        # Column 0 preserved, column 1 zeroed
        np.testing.assert_array_equal(result, [[1, 0], [3, 0]])

    def test_width_1_edge_case(self):
        """Width 1: midpoint=0, left filter keeps nothing, right filter keeps all."""
        # With width 1, midpoint = 0
        # left: result[:, 0:] = 0 -> zeros everything
        mock_left = MockAtlasS3Heatmap(
            hemisphere="left", orientation="frontal"
        )
        slice_data = np.array([[5]])

        result = mock_left._apply_hemisphere_filter(slice_data)
        # midpoint=0, so result[:, 0:] = 0 zeros the only column
        assert result[0, 0] == 0


class TestDataTypePreservation:
    """Tests for data type handling."""

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(np.int32, id="int32"),
            pytest.param(np.int64, id="int64"),
            pytest.param(np.float32, id="float32"),
            pytest.param(np.float64, id="float64"),
            pytest.param(np.uint8, id="uint8"),
        ],
    )
    def test_dtype_preserved(self, dtype):
        """Output array should preserve input dtype."""
        mock = MockAtlasS3Heatmap(hemisphere="left", orientation="frontal")
        slice_data = np.ones((10, 20), dtype=dtype)

        result = mock._apply_hemisphere_filter(slice_data)

        assert result.dtype == dtype


class TestWithRealAnnotationValues:
    """Tests simulating real atlas annotation IDs."""

    def test_preserves_region_ids_in_kept_hemisphere(self):
        """Real annotation IDs should be preserved in the kept hemisphere."""
        mock = MockAtlasS3Heatmap(hemisphere="left", orientation="frontal")
        # Simulate annotation IDs (0=background, nonzero=region IDs)
        slice_data = np.array(
            [
                [0, 100, 200, 0, 300, 400],
                [50, 150, 250, 350, 450, 550],
            ]
        )

        result = mock._apply_hemisphere_filter(slice_data)

        # Left half (columns 0-2) preserved
        np.testing.assert_array_equal(result[:, :3], slice_data[:, :3])
        # Right half (columns 3-5) zeroed
        np.testing.assert_array_equal(result[:, 3:], [[0, 0, 0], [0, 0, 0]])

    def test_all_zero_slice_unchanged(self):
        """Slice with no brain tissue (all zeros) should remain unchanged."""
        mock = MockAtlasS3Heatmap(hemisphere="left", orientation="frontal")
        slice_data = np.zeros((10, 20), dtype=int)

        result = mock._apply_hemisphere_filter(slice_data)

        np.testing.assert_array_equal(result, slice_data)
