"""Unit tests for AtlasS3Heatmap.validate_basic_params static method."""

import numpy as np
import pytest

from brainglobe_heatmap.heatmaps_atlas_s3 import (
    S3_ATLAS_MAPPING,
    AtlasS3Heatmap,
)


class TestValidateBasicParamsFormat:
    """Tests for format parameter validation."""

    def test_valid_2d_format_passes(self):
        """Valid '2D' format should not raise."""
        AtlasS3Heatmap.validate_basic_params(
            format="2D",
            position=1000,
            orientation="frontal",
            atlas_name="allen_mouse_25um",
        )

    @pytest.mark.parametrize(
        "invalid_format",
        [
            pytest.param("3D", id="3D_format"),
            pytest.param("2d", id="lowercase_2d"),
            pytest.param(" 2D ", id="whitespace_padded"),
            pytest.param("", id="empty_string"),
            pytest.param(None, id="none"),
            pytest.param(2, id="integer"),
        ],
    )
    def test_invalid_format_raises(self, invalid_format):
        """Invalid format values should raise ValueError."""
        with pytest.raises(ValueError, match="only format='2D' is supported"):
            AtlasS3Heatmap.validate_basic_params(
                format=invalid_format,
                position=1000,
                orientation="frontal",
                atlas_name="allen_mouse_25um",
            )


class TestValidateBasicParamsPosition:
    """Tests for position parameter validation."""

    @pytest.mark.parametrize(
        "valid_position",
        [
            pytest.param(0, id="zero_int"),
            pytest.param(1000, id="positive_int"),
            pytest.param(-50, id="negative_int"),
            pytest.param(100.5, id="positive_float"),
            pytest.param(-50.123, id="negative_float"),
            pytest.param(np.int32(100), id="numpy_int32"),
            pytest.param(np.int64(100), id="numpy_int64"),
            pytest.param(np.float32(100.5), id="numpy_float32"),
            pytest.param(np.float64(100.5), id="numpy_float64"),
        ],
    )
    def test_valid_position_types_pass(self, valid_position):
        """Valid numeric position types should not raise."""
        AtlasS3Heatmap.validate_basic_params(
            format="2D",
            position=valid_position,
            orientation="frontal",
            atlas_name="allen_mouse_25um",
        )

    @pytest.mark.parametrize(
        "invalid_position",
        [
            pytest.param("100", id="string_number"),
            pytest.param([100], id="list"),
            pytest.param((100,), id="tuple"),
            pytest.param(np.array([100]), id="numpy_array"),
            pytest.param(None, id="none"),
            pytest.param({"value": 100}, id="dict"),
        ],
    )
    def test_invalid_position_types_raise(self, invalid_position):
        """Non-numeric position types should raise ValueError."""
        with pytest.raises(
            ValueError, match="position must be a single numeric value"
        ):
            AtlasS3Heatmap.validate_basic_params(
                format="2D",
                position=invalid_position,
                orientation="frontal",
                atlas_name="allen_mouse_25um",
            )


class TestValidateBasicParamsOrientation:
    """Tests for orientation parameter validation."""

    @pytest.mark.parametrize(
        "valid_orientation",
        [
            pytest.param("frontal", id="frontal"),
            pytest.param("horizontal", id="horizontal"),
            pytest.param("sagittal", id="sagittal"),
        ],
    )
    def test_valid_orientations_pass(self, valid_orientation):
        """Valid orientation strings should not raise."""
        AtlasS3Heatmap.validate_basic_params(
            format="2D",
            position=1000,
            orientation=valid_orientation,
            atlas_name="allen_mouse_25um",
        )

    @pytest.mark.parametrize(
        "invalid_orientation",
        [
            pytest.param("Frontal", id="capitalized"),
            pytest.param("SAGITTAL", id="uppercase"),
            pytest.param("saggital", id="misspelled"),
            pytest.param("frontal_view", id="extra_suffix"),
            pytest.param(" frontal ", id="whitespace_padded"),
            pytest.param("", id="empty_string"),
        ],
    )
    def test_invalid_orientation_values_raise(self, invalid_orientation):
        """Invalid orientation strings should raise ValueError."""
        with pytest.raises(ValueError, match="orientation must be"):
            AtlasS3Heatmap.validate_basic_params(
                format="2D",
                position=1000,
                orientation=invalid_orientation,
                atlas_name="allen_mouse_25um",
            )

    @pytest.mark.parametrize(
        "non_string_orientation",
        [
            pytest.param(0, id="integer"),
            pytest.param(None, id="none"),
            pytest.param(["frontal"], id="list"),
            pytest.param((0, 0, 1), id="tuple_vector"),
        ],
    )
    def test_non_string_orientation_raises(self, non_string_orientation):
        """Non-string orientation types should raise ValueError."""
        with pytest.raises(ValueError, match="orientation must be"):
            AtlasS3Heatmap.validate_basic_params(
                format="2D",
                position=1000,
                orientation=non_string_orientation,
                atlas_name="allen_mouse_25um",
            )


class TestValidateBasicParamsAtlasName:
    """Tests for atlas_name parameter validation."""

    def test_none_atlas_name_passes(self):
        """None atlas_name should pass (defaults to allen_mouse_25um later)."""
        AtlasS3Heatmap.validate_basic_params(
            format="2D",
            position=1000,
            orientation="frontal",
            atlas_name=None,
        )

    @pytest.mark.parametrize(
        "valid_atlas",
        [
            pytest.param("allen_mouse_10um", id="allen_10um"),
            pytest.param("allen_mouse_25um", id="allen_25um"),
            pytest.param("allen_mouse_50um", id="allen_50um"),
            pytest.param("allen_mouse_100um", id="allen_100um"),
            pytest.param("kim_mouse_25um", id="kim_25um"),
            pytest.param("osten_mouse_50um", id="osten_50um"),
            pytest.param("csl_cat_500um", id="csl_cat"),
        ],
    )
    def test_valid_atlas_names_pass(self, valid_atlas):
        """Valid atlas names from S3_ATLAS_MAPPING should not raise."""
        AtlasS3Heatmap.validate_basic_params(
            format="2D",
            position=1000,
            orientation="frontal",
            atlas_name=valid_atlas,
        )

    @pytest.mark.parametrize(
        "invalid_atlas",
        [
            pytest.param("Allen_Mouse_25um", id="wrong_case"),
            pytest.param("allen_mouse_26um", id="typo_resolution"),
            pytest.param("monkey_atlas", id="nonexistent"),
            pytest.param("", id="empty_string"),
            pytest.param(" allen_mouse_25um ", id="whitespace_padded"),
        ],
    )
    def test_invalid_atlas_names_raise(self, invalid_atlas):
        """Invalid atlas names should raise ValueError with supported list."""
        with pytest.raises(
            ValueError, match="not supported for S3 atlas mode"
        ):
            AtlasS3Heatmap.validate_basic_params(
                format="2D",
                position=1000,
                orientation="frontal",
                atlas_name=invalid_atlas,
            )

    def test_error_message_lists_supported_atlases(self):
        """Error message should list all supported atlases."""
        with pytest.raises(ValueError) as exc_info:
            AtlasS3Heatmap.validate_basic_params(
                format="2D",
                position=1000,
                orientation="frontal",
                atlas_name="invalid_atlas",
            )
        error_message = str(exc_info.value)
        # Check that at least some known atlases are listed
        assert "allen_mouse_25um" in error_message
        assert "kim_mouse_25um" in error_message


class TestValidateBasicParamsAllValid:
    """Tests for complete valid parameter combinations."""

    def test_all_valid_params_passes(self):
        """All valid parameters should not raise any error."""
        AtlasS3Heatmap.validate_basic_params(
            format="2D",
            position=5000.5,
            orientation="horizontal",
            atlas_name="kim_mouse_50um",
        )

    def test_s3_atlas_mapping_coverage(self):
        """All atlases in S3_ATLAS_MAPPING should be valid."""
        for atlas_name in S3_ATLAS_MAPPING.keys():
            AtlasS3Heatmap.validate_basic_params(
                format="2D",
                position=1000,
                orientation="frontal",
                atlas_name=atlas_name,
            )
