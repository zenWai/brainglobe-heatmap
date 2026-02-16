import ast
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import ngff_zarr
import numpy as np
import pandas as pd
import rasterio.features
from brainglobe_atlasapi import BrainGlobeAtlas
from brainrender import settings
from matplotlib import pyplot as plt
from shapely.algorithms.polylabel import polylabel
from shapely.geometry import shape as shapely_shape
from vedo.colors import color_map as vedo_map_color

S3_BUCKET_STUB = "s3://brainglobe/atlas/{}"

# from atlas_name to S3 paths and pyramid level
S3_ATLAS_MAPPING = {
    # Allen mouse (4 resolutions: 10um, 25um, 50um, 100um)
    # Kim mouse (4 resolutions: 10um, 25um, 50um, 100um)
    # Osten mouse (4 resolutions: 10um, 25um, 50um, 100um)
    # Note: Uses kim terminology # TODO: fine?
    # Allen mouse with BlueBrain barrels (2 resolutions: 10um, 25um)
    # CSL cat (1 resolution: 500um only)
    "allen_mouse_10um": {
        "annotation": "annotation-sets/allen-adult-mouse-annotation/2017/annotation.ome.zarr",
        "terminology": "terminologies/allen-adult-mouse-terminology/2017/terminology.csv",
        "pyramid_level": 0,
    },
    "allen_mouse_25um": {
        "annotation": "annotation-sets/allen-adult-mouse-annotation/2017/annotation.ome.zarr",
        "terminology": "terminologies/allen-adult-mouse-terminology/2017/terminology.csv",
        "pyramid_level": 1,
    },
    "allen_mouse_50um": {
        "annotation": "annotation-sets/allen-adult-mouse-annotation/2017/annotation.ome.zarr",
        "terminology": "terminologies/allen-adult-mouse-terminology/2017/terminology.csv",
        "pyramid_level": 2,
    },
    "allen_mouse_100um": {
        "annotation": "annotation-sets/allen-adult-mouse-annotation/2017/annotation.ome.zarr",
        "terminology": "terminologies/allen-adult-mouse-terminology/2017/terminology.csv",
        "pyramid_level": 3,
    },
    "kim_mouse_10um": {
        "annotation": "annotation-sets/kim-adult-mouse-annotation/2017/annotation.ome.zarr",
        "terminology": "terminologies/kim-adult-mouse-terminology/2017/terminology.csv",
        "pyramid_level": 0,
    },
    "kim_mouse_25um": {
        "annotation": "annotation-sets/kim-adult-mouse-annotation/2017/annotation.ome.zarr",
        "terminology": "terminologies/kim-adult-mouse-terminology/2017/terminology.csv",
        "pyramid_level": 1,
    },
    "kim_mouse_50um": {
        "annotation": "annotation-sets/kim-adult-mouse-annotation/2017/annotation.ome.zarr",
        "terminology": "terminologies/kim-adult-mouse-terminology/2017/terminology.csv",
        "pyramid_level": 2,
    },
    "kim_mouse_100um": {
        "annotation": "annotation-sets/kim-adult-mouse-annotation/2017/annotation.ome.zarr",
        "terminology": "terminologies/kim-adult-mouse-terminology/2017/terminology.csv",
        "pyramid_level": 3,
    },
    "osten_mouse_10um": {
        "annotation": "annotation-sets/osten-adult-mouse-annotation/1_1/annotation.ome.zarr",
        "terminology": "terminologies/kim-adult-mouse-terminology/2017/terminology.csv",
        "pyramid_level": 0,
    },
    "osten_mouse_25um": {
        "annotation": "annotation-sets/osten-adult-mouse-annotation/1_1/annotation.ome.zarr",
        "terminology": "terminologies/kim-adult-mouse-terminology/2017/terminology.csv",
        "pyramid_level": 1,
    },
    "osten_mouse_50um": {
        "annotation": "annotation-sets/osten-adult-mouse-annotation/1_1/annotation.ome.zarr",
        "terminology": "terminologies/kim-adult-mouse-terminology/2017/terminology.csv",
        "pyramid_level": 2,
    },
    "osten_mouse_100um": {
        "annotation": "annotation-sets/osten-adult-mouse-annotation/1_1/annotation.ome.zarr",
        "terminology": "terminologies/kim-adult-mouse-terminology/2017/terminology.csv",
        "pyramid_level": 3,
    },
    "allen_mouse_bluebrain_barrels_10um": {
        "annotation": "annotation-sets/allen_bluebrain_barrels-adult-mouse-annotation/1_0/annotation.ome.zarr",
        "terminology": "terminologies/allen_bluebrain_barrels-adult-mouse-terminology/1_0/terminology.csv",
        "pyramid_level": 0,
    },
    "allen_mouse_bluebrain_barrels_25um": {
        "annotation": "annotation-sets/allen_bluebrain_barrels-adult-mouse-annotation/1_0/annotation.ome.zarr",
        "terminology": "terminologies/allen_bluebrain_barrels-adult-mouse-terminology/1_0/terminology.csv",
        "pyramid_level": 1,
    },
    "csl_cat_500um": {
        "annotation": "annotation-sets/csl-adult-cat-annotation/2017/annotation.ome.zarr",
        "terminology": "terminologies/csl-adult-cat-terminology/2017/terminology.csv",
        "pyramid_level": 0,
    },
}
ORIENTATION_TO_AXIS = {
    "frontal": 0,  # slice_data[z, :, :]
    "horizontal": 1,  # slice_data[:, y, :]
    "sagittal": 2,  # slice_data[:, :, x]
}


class AtlasS3Heatmap:
    """
    For S3 atlas-based heatmap rendering.
    Used internally by Heatmap when use_atlas=True.

    2D heatmap visualization using annotation zarr data and terminology CSV files
    from brainglobe s3 bucket.

    Bypasses the brainrender/Slicer pipeline.
    """

    # Pinpoint data position slide with website slide
    KIM_ATLAS_OFFSET_MICRONS = 800
    KIM_WEBSITE_SLICE_SPACING = 100

    @staticmethod
    def kim_website_slice_to_position(
        website_slice: int, verbose: bool = True
    ) -> float:
        """
        Convert Kim 3D website slice number to position in microns.
        """
        spacing = AtlasS3Heatmap.KIM_WEBSITE_SLICE_SPACING
        offset = AtlasS3Heatmap.KIM_ATLAS_OFFSET_MICRONS

        position_start = website_slice * spacing + offset
        position_end = position_start + spacing - 1

        if verbose:
            print(
                f"Kim website slice {website_slice} -> position {position_start}um "
                f"(valid range: {position_start}um to {position_end}um)"
            )

        return float(position_start)

    @staticmethod
    def position_to_kim_website_slice(position: float) -> int:
        """
        Convert position in microns to Kim 3D website slice number.
        """
        website_slice = int(
            (position - AtlasS3Heatmap.KIM_ATLAS_OFFSET_MICRONS)
            / AtlasS3Heatmap.KIM_WEBSITE_SLICE_SPACING
        )
        return website_slice

    @staticmethod
    def validate_basic_params(
        format, position, orientation, atlas_name
    ) -> None:
        """Validate basic parameters before loading data.
        NOT SUPPORTED YET
        - 3D
        - plane angle (position float, orientation float)
        - atlas -> S3_ATLAS_MAPPING only
        """

        if format != "2D":
            raise ValueError(
                "When use_atlas=True, only format='2D' is supported"
            )

        # Position must be single numeric value
        if not isinstance(position, (int, float, np.number)):
            raise ValueError(
                "When use_atlas=True, position must be a single numeric value"
            )

        # Orientation must be a string
        if not isinstance(orientation, str):
            raise ValueError(
                "When use_atlas=True, orientation must be "
                "'frontal', 'sagittal', or 'horizontal'"
            )

        if orientation not in ORIENTATION_TO_AXIS:
            raise ValueError(
                "When use_atlas=True, orientation must be "
                "'frontal', 'sagittal', or 'horizontal'"
            )

        # Atlas name must be in the supported mapping (None defaults to allen_mouse_25um)
        if atlas_name is not None and atlas_name not in S3_ATLAS_MAPPING:
            supported = ", ".join(sorted(S3_ATLAS_MAPPING.keys()))
            raise ValueError(
                f"Atlas '{atlas_name}' not supported for S3 atlas mode. "
                f"Supported atlases: {supported}"
            )

    @staticmethod
    def load_atlas_data(
        atlas_name: str,
    ) -> tuple[pd.DataFrame, ngff_zarr.ngff_image]:
        """Load terminology CSV and annotation zarr from S3 with caching."""
        atlas_mapping = S3_ATLAS_MAPPING[atlas_name]

        # Get paths and pyramid level from the mapping
        pyramid_level = atlas_mapping["pyramid_level"]
        terminology_uri = S3_BUCKET_STUB.format(atlas_mapping["terminology"])
        annotation_uri = S3_BUCKET_STUB.format(atlas_mapping["annotation"])

        # Load terminology
        terminology_df = pd.read_csv(
            terminology_uri, storage_options={"anon": True}
        )
        # Load annotation zarr
        annotations = ngff_zarr.from_ngff_zarr(
            annotation_uri, storage_options={"anon": True}
        )
        annotation_image = annotations.images[pyramid_level]

        return terminology_df, annotation_image

    @staticmethod
    def _find_annotation_position(mask: np.ndarray) -> Tuple[float, float]:
        """
        Find optimal annotation position inside a mask using rasterio + polylabel.
        """
        mask_uint8 = mask.astype(np.uint8)
        shapes_gen = rasterio.features.shapes(mask_uint8, mask=mask)
        polygons = [shapely_shape(geom) for geom, value in shapes_gen]

        # Get largest polygon
        geometry = max(polygons, key=lambda p: p.area)

        label_position = polylabel(geometry, tolerance=1.0)  # type: ignore[arg-type]
        return label_position.x, label_position.y

    def __init__(
        self,
        values: Dict,
        position: float,
        orientation: str,
        atlas_name: str | None,
        hemisphere: str,
        cmap: str,
        vmin: Optional[float],
        vmax: Optional[float],
        annotate_regions: Optional[Union[bool, List[str], Dict]],
        annotate_text_options_2d: Optional[Dict],
        use_reference: bool = False,
    ):
        """
        Initialize AtlasS3Heatmap.

        Parameters
        ----------
        values : Dict
            Dictionary with brain region acronyms as keys and values.
        position : float
            Position along the slicing axis in microns.
        orientation : str
            One of 'frontal', 'sagittal', or 'horizontal'.
        atlas_name : str | None
            Name of the atlas (e.g., 'allen_mouse_25um').
        hemisphere : str
            'left', 'right', or 'both'.
        cmap : str
            Matplotlib colormap name.
        vmin : Optional[float]
            Minimum value for colormap.
        vmax : Optional[float]
            Maximum value for colormap.
        annotate_regions : Optional[Union[bool, List[str], Dict]]
            Region annotation settings.
        annotate_text_options_2d : Optional[Dict]
            Text options for annotations.
        use_reference : bool
            If True, draw the brain reference image as background
            instead of the solid-color brain root.
            Loads reference.tiff from local ~/.brainglobe atlas.
        """
        self.values = values
        self.position = position
        self.orientation = orientation
        self.atlas_name = atlas_name or "allen_mouse_25um"
        self.hemisphere = hemisphere
        self.annotate_regions = annotate_regions
        self.annotate_text_options_2d = annotate_text_options_2d
        self.use_reference = use_reference

        # Initialize: load data, validate regions exist on atlas, prepare colors
        self.terminology_df, self.annotation_image = self.load_atlas_data(
            self.atlas_name
        )
        self._validate_region_values_exist()
        self.colors, self.vmin, self.vmax = self._prepare_colors(
            cmap, vmin, vmax
        )

        if self.use_reference:
            self.reference_data = self._load_reference()

    def _validate_region_values_exist(self) -> None:
        """Validate region names and values after loading terminology."""
        # Validate region regions exist in terminology
        valid_regions = self.terminology_df["abbreviation"].values
        not_valid_regions = [
            region
            for region in self.values.keys()
            if region not in valid_regions
        ]
        if not_valid_regions:
            raise ValueError(f"Region(s) {not_valid_regions} not recognized")

        # Validate values are numeric
        for k, v in self.values.items():
            if not isinstance(v, (float, int)):
                raise ValueError(
                    f"Heatmap values should be floats, "
                    f'not: {type(v)} for entry "{k}"'
                )

    def _prepare_colors(
        self,
        cmap: str,
        vmin: Optional[float],
        vmax: Optional[float],
    ) -> tuple[dict[str, Union[list, str]], float, float]:
        """Prepare color mapping for regions."""
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
        colors: dict[str, Union[list, str]] = {
            r: list(
                vedo_map_color(v, name=cmap, vmin=final_vmin, vmax=final_vmax)
            )
            for r, v in self.values.items()
        }
        colors["root"] = settings.ROOT_COLOR

        return colors, final_vmin, final_vmax

    def _load_reference(self) -> np.ndarray:
        """Load the reference image from local ~/.brainglobe atlas."""
        bg_atlas = BrainGlobeAtlas(self.atlas_name)
        return bg_atlas.reference

    def _get_reference_slice(self) -> np.ndarray:
        """Extract 2D slice from the reference image."""
        shape = self.reference_data.shape
        axis = ORIENTATION_TO_AXIS[self.orientation]

        pos = self._microns_to_voxels(self.position)
        pos = min(pos, shape[axis] - 1)
        pos = max(0, pos)

        if self.orientation == "frontal":
            return self.reference_data[pos, :, :]
        elif self.orientation == "horizontal":
            return self.reference_data[:, pos, :]
        else:  # sagittal
            return self.reference_data[:, :, pos].T

    def _microns_to_voxels(self, position_microns: float) -> int:
        """Convert position from microns to voxel coordinates."""
        scale_mm = self.annotation_image.scale
        scale_microns = {k: v * 1000 for k, v in scale_mm.items()}

        axis_to_dim = {0: "z", 1: "y", 2: "x"}
        axis = ORIENTATION_TO_AXIS[self.orientation]
        dim = axis_to_dim[axis]

        voxel_pos = int(position_microns / scale_microns[dim])
        return voxel_pos

    def _voxels_to_microns(self, voxel_count: int, axis: str) -> float:
        """Convert voxel count to microns for a given axis."""
        scale_mm = self.annotation_image.scale
        return voxel_count * scale_mm[axis] * 1000

    def _get_slice_extent_centered(
        self, h, w
    ) -> Tuple[float, float, float, float]:
        """Get centered-at-zero extent for imshow based on orientation."""

        # Map orientation to physical axes for (height, width)
        orientation_axes = {
            "frontal": ("y", "x"),  # rows=Y, cols=X
            "horizontal": ("z", "x"),  # rows=Z, cols=X
            "sagittal": ("y", "z"),  # rows=Y, cols=Z (after transpose)
        }

        h_axis, w_axis = orientation_axes[self.orientation]
        y_size = self._voxels_to_microns(h, h_axis)
        x_size = self._voxels_to_microns(w, w_axis)

        return -x_size / 2, x_size / 2, -y_size / 2, y_size / 2

    def _pixel_to_microns_centered(
        self, pixel_pos: Tuple[float, float], shape: Tuple[int, int]
    ) -> Tuple[float, float]:
        """Convert pixel position to centered micron coordinates."""
        h, w = shape
        x_pixel, y_pixel = pixel_pos

        orientation_axes = {
            "frontal": ("y", "x"),
            "horizontal": ("z", "x"),
            "sagittal": ("y", "z"),
        }
        h_axis, w_axis = orientation_axes[self.orientation]

        # Convert to microns
        x_microns = self._voxels_to_microns(int(x_pixel), w_axis)
        y_microns = self._voxels_to_microns(int(y_pixel), h_axis)

        # Center (subtract half of total size)
        x_centered = x_microns - self._voxels_to_microns(w, w_axis) / 2
        # Flip Y: pixel y=0 (top) → positive microns, pixel y=h (bottom) → negative
        y_centered = self._voxels_to_microns(h, h_axis) / 2 - y_microns

        return x_centered, y_centered

    def _get_atlas_slice(self) -> np.ndarray:
        """Extract 2D slice based on orientation and position."""
        shape = self.annotation_image.data.shape
        axis = ORIENTATION_TO_AXIS[self.orientation]

        pos = self._microns_to_voxels(self.position)
        pos = min(pos, shape[axis] - 1)
        pos = max(0, pos)

        if self.orientation == "frontal":
            return self.annotation_image.data[pos, :, :].compute()
        elif self.orientation == "horizontal":
            return self.annotation_image.data[:, pos, :].compute()
        else:  # sagittal
            slice_data = self.annotation_image.data[:, :, pos].compute()
            return slice_data.T

    def _apply_hemisphere_filter(self, slice_data: np.ndarray) -> np.ndarray:
        """Mask out opposite hemisphere for 'left' or 'right' modes."""
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

    def _get_region_ids(self, abbreviation: str) -> List[int]:
        """Get annotation IDs for region, optionally including descendants."""
        exact_match = self.terminology_df[
            self.terminology_df["abbreviation"] == abbreviation
        ]

        if len(exact_match) == 0:
            return []

        target_id = int(exact_match["identifier"].values[0])

        def has_ancestor(path_str, ancestor_id):
            path_list = ast.literal_eval(str(path_str))
            return ancestor_id in path_list

        descendants = self.terminology_df[
            self.terminology_df["root_identifier_path"].apply(
                lambda x: has_ancestor(x, target_id)
            )
        ]

        return descendants["annotation_value"].tolist()

    def _get_hierarchy_depth(self, abbreviation: str) -> int:
        """Get hierarchy depth of a region for sorting (parents first)."""
        exact_match = self.terminology_df[
            self.terminology_df["abbreviation"] == abbreviation
        ]
        if len(exact_match) == 0:
            return 0
        path_str = exact_match["root_identifier_path"].values[0]
        return len(ast.literal_eval(str(path_str)))

    def _get_visible_regions(
        self, slice_data: np.ndarray
    ) -> List[Tuple[str, List[int]]]:
        """Get user regions that are visible in the slice."""
        unique_ids_in_slice = set(np.unique(slice_data))

        visible_regions = []
        for region in self.values.keys():
            region_ids = self._get_region_ids(region)

            if any(rid in unique_ids_in_slice for rid in region_ids):
                visible_regions.append((region, region_ids))

        return visible_regions

    def _get_annotation_text(self, region_name: str) -> Union[None, str]:
        """
        Gets the annotation text for a region if it should be annotated

        Returns
        -------
        None or str
            None if the region should not be annotated.

        Notes
        -----
        The behavior depends on the type of self.annotate_regions:
        - If bool: All regions except "root" are annotated when True
        - If list: Only regions in the list are annotated except "root"
        - If dict: Only regions in the dict keys are annotated,
          using dict values as display text
        """
        if region_name == "root":
            return None

        should_annotate = (
            (isinstance(self.annotate_regions, bool) and self.annotate_regions)
            or (
                isinstance(self.annotate_regions, list)
                and region_name in self.annotate_regions
            )
            or (
                isinstance(self.annotate_regions, dict)
                and region_name in self.annotate_regions.keys()
            )
        )

        if not should_annotate:
            return None

        # Determine what text to use for annotation
        if isinstance(self.annotate_regions, dict):
            return str(self.annotate_regions[region_name])

        return region_name

    def atlas_s3_render_to_axes(
        self,
        ax: plt.Axes,
        contour_color: str = "black",
        contour_width: float = 0.5,
    ) -> None:
        """Render the heatmap to the given matplotlib axes."""

        # Get slice data
        slice_data = self._get_atlas_slice()
        slice_data = self._apply_hemisphere_filter(slice_data)

        # Get only the regions visible in this slice
        visible_regions = self._get_visible_regions(slice_data)

        # Sort by hierarchy depth (parents first, children overwrite)
        visible_regions.sort(key=lambda r: self._get_hierarchy_depth(r[0]))

        # Create RGBA canvas (white background)
        canvas = np.ones((*slice_data.shape, 4), dtype=np.float32)

        # Brain mask
        brain_mask = slice_data > 0

        if self.use_reference:
            ref_slice = self._get_reference_slice()
            # Normalize uint16 to [0, 1]
            ref_max = ref_slice.max()
            if ref_max > 0:
                ref_norm = ref_slice.astype(np.float32) / ref_max
            else:
                ref_norm = ref_slice.astype(np.float32)
            # Apply as grayscale background where brain exists
            canvas[brain_mask, 0] = ref_norm[brain_mask]
            canvas[brain_mask, 1] = ref_norm[brain_mask]
            canvas[brain_mask, 2] = ref_norm[brain_mask]
            canvas[brain_mask, 3] = 1.0
        else:
            canvas[brain_mask] = mcolors.to_rgba(
                self.colors["root"], alpha=settings.ROOT_ALPHA
            )

        # Store masks for contours and annotations
        region_masks = []

        for region, region_ids in visible_regions:
            mask = np.isin(slice_data, region_ids)
            canvas[mask] = mcolors.to_rgba(self.colors[region])
            region_masks.append((region, mask))

        # voxels to microns for axis, contours, annotations
        h, w = slice_data.shape
        # Y grow upwards, X grow right, 0-centered
        axis_extent = self._get_slice_extent_centered(h, w)
        # Contour have different indexes wtf
        contour_extent = (
            axis_extent[0],
            axis_extent[1],
            axis_extent[3],
            axis_extent[2],
        )

        ax.imshow(canvas, extent=axis_extent)

        # Draw contours for each region
        for region, mask in region_masks:
            ax.contour(
                mask.astype(float),
                levels=[0.5],
                colors=[contour_color],
                linewidths=[contour_width],
                extent=contour_extent,
            )

            # Handle annotations
            display_text = self._get_annotation_text(region)
            if display_text is not None:
                pixel_pos = self._find_annotation_position(mask)
                micron_pos = self._pixel_to_microns_centered(
                    pixel_pos, slice_data.shape
                )
                ax.annotate(
                    display_text,
                    xy=micron_pos,
                    ha="center",
                    va="center",
                    **(self.annotate_text_options_2d or {}),
                )

        # Fit axis limits to brain content
        rows = np.any(brain_mask, axis=1)
        cols = np.any(brain_mask, axis=0)
        y_min_px, y_max_px = np.where(rows)[0][[0, -1]]
        x_min_px, x_max_px = np.where(cols)[0][[0, -1]]

        x_min_um, y_min_um = self._pixel_to_microns_centered(
            (x_min_px, y_max_px), slice_data.shape
        )
        x_max_um, y_max_um = self._pixel_to_microns_centered(
            (x_max_px, y_min_px), slice_data.shape
        )

        # Add 3% padding
        x_padding = (x_max_um - x_min_um) * 0.04
        y_padding = (y_max_um - y_min_um) * 0.04
        ax.set_xlim(x_min_um - x_padding, x_max_um + x_padding)
        ax.set_ylim(y_min_um - y_padding, y_max_um + y_padding)
        ax.set_aspect("equal")
        # flow continues on Heatmaps.plot_subplot
