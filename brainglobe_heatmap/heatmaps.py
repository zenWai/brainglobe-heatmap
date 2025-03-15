from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from brainrender import Scene, cameras, settings
from brainrender.actor import Actor
from brainrender.atlas import Atlas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myterial import grey_darker
from shapely import Polygon
from shapely.algorithms.polylabel import polylabel
from shapely.geometry.multipolygon import MultiPolygon
from vedo import Point
from vedo.colors import color_map as map_color

from brainglobe_heatmap.slicer import Slicer

# Set settings for heatmap visualization
settings.SHOW_AXES = False
settings.SHADER_STYLE = "cartoon"
settings.ROOT_ALPHA = 0.3
settings.ROOT_COLOR = grey_darker

# Set settings for transparent background
# vedo for transparent bg
# settings.vsettings.screenshot_transparent_background = True

# This needs to be false for transparent bg
# settings.vsettings.use_fxaa = False


def check_values(values: dict, atlas: Atlas) -> Tuple[float, float]:
    """
    Checks that the passed heatmap values meet two criteria:
        - keys should be acronyms of brainregions
        - values should be numbers
    """
    for k, v in values.items():
        if not isinstance(v, (float, int)):
            raise ValueError(
                f"Heatmap values should be floats, "
                f'not: {type(v)} for entry "{k}"'
            )

        if k not in atlas.lookup_df.acronym.values:
            raise ValueError(f'Region name "{k}" not recognized')

    not_nan = [v for v in values.values() if not np.isnan(v)]
    if len(not_nan) == 0:
        return np.nan, np.nan
    vmax, vmin = max(not_nan), min(not_nan)
    return vmax, vmin


def find_annotation_position_inside_polygon(
    polygon_vertices: np.ndarray,
) -> Union[Tuple[float, float], None]:
    """
    Finds a suitable point for annotation within a polygon.

    Returns
    -------
    Tuple[float, float] or None
        A tuple (x, y) representing the point
        None if not enough vertices to form a valid polygon.

    Notes
    -----
    2D polygons only
    Edge cases:
    - Requires at least 4 vertices (< 4 returns None)
    - For invalid polygons, reconstructs the polygon using buffer(0),
      this resolves e.g., self-intersections
    - For some types of invalid geometries,
      buffer(0) may create a shapely MultiPolygon object by
      splitting self-intersecting areas into separate valid polygons.
      When this happens, the function gets the largest polygon by area.
    - Uses Shapely's polylabel algorithm with a tolerance of 0.1
      that accepts a polygon after edge cases resolved.
    """
    if polygon_vertices.shape[0] < 4:
        return None
    polygon = Polygon(polygon_vertices.tolist())

    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    if polygon.geom_type == "MultiPolygon" and isinstance(
        polygon, MultiPolygon
    ):
        polygon = max(polygon.geoms, key=lambda p: p.area)

    label_position = polylabel(polygon, tolerance=0.1)
    return label_position.x, label_position.y


# TODO: test colors
def export_powerpoint_compatible_glb(self, path):
    """
    Export a PowerPoint-compatible GLB file using PyVista and Trimesh.
    """
    from vedo import Plotter

    if path.suffix != ".glb":
        path = path.with_suffix(".glb")

    # Create a vedo Plotter and add the clean renderables
    plt_obj = Plotter(interactive=False, offscreen=True)
    plt_obj.add(self.scene.clean_renderables)
    plt_obj = plt_obj.show(interactive=False)

    # pyvista to process the meshes Trimesh to optimize
    import pyvista as pv
    import trimesh

    all_vertices = []
    all_faces = []
    all_colors = []
    vertex_offset = 0

    for actor in plt_obj.get_actors():
        if hasattr(actor, "GetProperty"):
            prop = actor.GetProperty()
            opacity = prop.GetOpacity()

            # Skip actors that are not fully opaque
            # OBS: brainrender root default opacity is < 1.0
            if opacity < 1.0:
                continue

            if hasattr(actor, "GetMapper") and actor.GetMapper():
                mapper = actor.GetMapper()

                # Process the mesh data if available
                if mapper.GetInput():
                    # Convert VTK data to PyVista mesh
                    pv_mesh = pv.wrap(mapper.GetInput())

                    # Extract face and vertex data
                    vertices = pv_mesh.points
                    faces = []

                    if hasattr(pv_mesh, "faces") and len(pv_mesh.faces) > 0:
                        # PyVista faces are stored with their size
                        # so a triangle is [3, v1, v2, v3]
                        # We need to convert to just [v1, v2, v3]
                        pv_faces = pv_mesh.faces.reshape(
                            (-1, 4)
                        )  # First value is number of points per face (3)
                        for face in pv_faces:
                            if face[0] == 3:
                                faces.append([face[1], face[2], face[3]])

                    # Only add to combined mesh if we have vertices and faces
                    if len(vertices) > 0 and len(faces) > 0:
                        # Get color for this mesh
                        color = prop.GetColor()  # RGB in 0-1 range

                        # Flip the model along y-axis
                        flipped_vertices = vertices.copy()
                        flipped_vertices[:, 1] = -flipped_vertices[:, 1]

                        # Add vertices to the combined list
                        all_vertices.append(flipped_vertices)

                        # Adjust face indices to account for the offset
                        offset_faces = np.array(faces) + vertex_offset
                        all_faces.append(offset_faces)

                        # Update vertex offset for the next mesh
                        vertex_offset += len(flipped_vertices)

                        # Add colors for all faces
                        mesh_colors = np.tile(
                            np.array([color[0], color[1], color[2], 1.0])
                            * 255,
                            (len(faces), 1),
                        ).astype(np.uint8)
                        all_colors.append(mesh_colors)

    if all_vertices and all_faces:
        # Combine all vertices and faces
        combined_vertices = np.vstack(all_vertices)
        combined_faces = np.vstack(all_faces)
        combined_colors = np.vstack(all_colors)

        # Create a single trimesh helps with powerpoint compatibility
        combined_mesh = trimesh.Trimesh(
            vertices=combined_vertices,
            faces=combined_faces,
            face_colors=combined_colors,
        )

        trimesh_scene = trimesh.Scene()
        trimesh_scene.add_geometry(combined_mesh, geom_name="brain_combined")

        # Export to GLB format
        trimesh_scene.export(str(path), file_type="glb")

        print(f"Successfully exported PowerPoint-compatible GLB to {path}")
        return str(path)

    return None


class Heatmap:
    def __init__(
        self,
        values: Dict,
        position: Union[list, tuple, np.ndarray, float],
        orientation: Union[str, tuple] = "frontal",
        hemisphere: str = "both",
        title: Optional[str] = None,
        cmap: str = "Reds",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        format: str = "3D",  # 3D -> brainrender, 2D -> matplotlib
        # brainrender, 3D HM specific
        thickness: float = 10,
        interactive: bool = True,
        zoom: Optional[float] = None,
        atlas_name: Optional[str] = None,
        label_regions: Optional[bool] = False,
        annotate_regions: Optional[Union[bool, List[str], Dict]] = False,
        annotate_less_clutter=False,
        annotate_text_options: Optional[Dict] = None,
        check_latest: bool = True,
        tight_layout_2d: bool = False,
        **kwargs,
    ):
        """
        Creates a heatmap visualization of the provided values in 3D or 2D
        using brainrender or matplotlib in the specified atlas.

        Parameters
        ----------
        values : dict
            Dictionary with brain regions acronyms as keys and
            magnitudes as the values.
        position : list, tuple, np.ndarray, float
            Position of the plane in the atlas.
        orientation : str or tuple, optional
            Orientation of the plane in the atlas. Either, "frontal",
            "sagittal", "horizontal" or a tuple with the normal vector.
            Default is "frontal".
        hemisphere : str, optional
            Hemisphere to display the heatmap. Default is "both".
        title : str, optional
            Title of the heatmap. Default is None.
        cmap : str, optional
            Colormap to use for the heatmap. Default is "Reds".
        vmin : float, optional
            Minimum value for the colormap. Default is None.
        vmax : float, optional
            Maximum value for the colormap. Default is None.
        format : str, optional
            Format of the heatmap visualization.
            "3D" for brainrender or "2D" for matplotlib. Default is "3D".
        thickness : float, optional
            Thickness of the slicing plane in the brainrender scene.
            Default is 10.
        interactive : bool, optional
            If True, the brainrender scene is interactive. Default is True.
        zoom : float, optional
            Zoom level for the brainrender scene. Default is None.
        atlas_name : str, optional
            Name of the atlas to use for the heatmap.
            If None allen_mouse_25um is used. Default is None.
        label_regions : bool, optional
            If True, labels the regions on the colorbar (only valid in 2D).
            Default is False.
        annotate_regions :
            bool, List[str], Dict[str, Union[str, float, int]], optional
            Controls region annotation in 2D format.
            If True, annotates all regions with their names.
            If a list, annotates only the specified regions.
            If a dict, uses custom text/values for annotations.
            Default is False.
        annotate_less_clutter :
            bool
            If True, annotate only one segment per brain region
                typically the largest segment—to reduce clutter.
            If False, annotate every segment with its region name.
        annotate_text_options : dict, optional
            Options for customizing region annotations text.
            Default is None
        check_latest : bool, optional
            Check for the latest version of the atlas. Default is True.
        """
        # store arguments
        self.values = values
        self.format = format
        self.orientation = orientation
        self.hemisphere = hemisphere
        self.interactive = interactive
        self.zoom = zoom
        self.title = title
        self.cmap = cmap
        self.label_regions = label_regions
        self.annotate_regions = annotate_regions
        self.annotate_less_clutter = annotate_less_clutter
        self.annotate_text_options = annotate_text_options
        self.tight_layout_2d = tight_layout_2d

        # create a scene
        self.scene = Scene(
            atlas_name=atlas_name,
            title=title,
            title_color=grey_darker,
            check_latest=check_latest,
            **kwargs,
        )

        # prep colors range
        self.prepare_colors(values, cmap, vmin, vmax)

        # add regions to the brainrender scene
        self.scene.add_brain_region(*self.values.keys(), hemisphere=hemisphere)

        self.regions_meshes = [
            r
            for r in self.scene.get_actors(br_class="brain region")
            if r.name != "root"
        ]

        # prepare slicer object
        self.slicer = Slicer(position, orientation, thickness, self.scene.root)

    def prepare_colors(
        self,
        values: dict,
        cmap: str,
        vmin: Optional[float],
        vmax: Optional[float],
    ):
        # get brain regions colors
        _vmax, _vmin = check_values(values, self.scene.atlas)
        if _vmax == _vmin:
            _vmin = _vmax * 0.5

        vmin = vmin if vmin == 0 or vmin else _vmin
        vmax = vmax if vmax == 0 or vmax else _vmax
        self.vmin, self.vmax = vmin, vmax

        self.colors = {
            r: list(map_color(v, name=cmap, vmin=vmin, vmax=vmax))
            for r, v in values.items()
        }
        self.colors["root"] = settings.ROOT_COLOR

    def get_region_annotation_text(self, region_name: str) -> Union[None, str]:
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

    def _get_optimal_label_position(self, mesh_intersection, mesh_center):
        """Helper function to find the optimal label position."""
        # Split the intersection into connected pieces
        pieces = mesh_intersection.split()

        # Sort pieces by size (largest first)
        pieces = sorted(pieces, key=lambda p: len(p.vertices), reverse=True)

        # Check each piece for a valid label position
        for piece in pieces:
            points_3d = piece.join(reset=True).vertices

            # Skip if we don't have enough points for a polygon
            if len(points_3d) < 4:
                continue

            # Project 3D points to 2D in the plane's coordinate system
            points_2d = self.slicer.plane0.p3_to_p2(points_3d)

            # Find the optimal position for the label
            optimal_pos_2d = find_annotation_position_inside_polygon(points_2d)

            if optimal_pos_2d is None:
                continue

            # Convert the 2D optimal position back to 3D
            optimal_pos_3d = self.slicer.plane0.center + (
                optimal_pos_2d[0] * self.slicer.plane0.u
                + optimal_pos_2d[1] * self.slicer.plane0.v
            )

            # Check if position is in correct hemisphere
            if not hasattr(self, "hemisphere") or self.hemisphere == "both":
                return optimal_pos_3d
            elif (
                self.hemisphere == "left"
                and optimal_pos_3d[2] > mesh_center[2]
            ) or (
                self.hemisphere == "right"
                and optimal_pos_3d[2] < mesh_center[2]
            ):
                return optimal_pos_3d

        return None

    def show(self, **kwargs) -> Union[Scene, plt.Figure]:
        """
        Creates a 2D plot or 3D rendering of the heatmap
        """
        if self.format == "3D":
            self.slicer.slice_scene(self.scene, self.regions_meshes)
            view = self.render(**kwargs)
        else:
            view = self.plot(**kwargs)
        return view

    def render(self, camera=None, **kwargs) -> Scene:
        """
        Renders the heatmap visualization as a 3D scene in brainrender.

        Parameters:
        ----------
        camera : str or dict, optional
            The `brainrender` camera to render the scene.
            If not provided, `self.orientation` is used.
        Returns:
        -------
        scene : Scene
            The rendered 3D scene.
        """

        # Set brain regions colors and add annotations where needed
        for region, color in self.colors.items():
            if region == "root":
                continue

            region_actors = self.scene.get_actors(
                br_class="brain region", name=region
            )
            if not region_actors:
                continue

            region_actor = region_actors[0]
            region_actor.color(color)

            # TODO: think on annotate html and glb / implement slice
            if not kwargs.get("export_html") or not kwargs.get("export_glb"):
                # Check if this region should be annotated
                display_text = self.get_region_annotation_text(
                    region_actor.name
                )
                if display_text is None:
                    continue

                # Get the region's intersection with the plane
                mesh_intersection = self.slicer.plane0.intersect_with(
                    region_actor.mesh
                )
                if (
                    not mesh_intersection
                    or len(mesh_intersection.vertices) < 4
                ):
                    continue

                # Get mesh center for hemisphere filtering
                mesh_center = (
                    self.scene.root.mesh.bounds().reshape((3, 2)).mean(axis=1)
                    if hasattr(self.scene.atlas, "metadata")
                    and self.scene.atlas.metadata.get("symmetric")
                    else self.scene.root.mesh.center_of_mass()
                )

                # Get optimal label position from largest valid piece
                optimal_pos_3d = self._get_optimal_label_position(
                    mesh_intersection, mesh_center
                )

                if optimal_pos_3d is None:
                    continue

                # Create and add the label
                label_actor = Actor(
                    Point(optimal_pos_3d, r=0.01).alpha(0),
                    name=f"{region_actor.name}_label",
                    br_class="brain region annotation",
                )
                self.scene.add(label_actor)
                self.scene.add_label(
                    actor=label_actor,
                    label=display_text,
                    size=300,
                    radius=100,
                    yoffset=100,
                )

        if camera is None:
            # set camera position and render
            if isinstance(self.orientation, str):
                if self.orientation == "sagittal":
                    camera = cameras.sagittal_camera2
                elif self.orientation == "horizontal":
                    camera = "top"
                else:
                    camera = self.orientation
            else:
                self.orientation = np.array(self.orientation)
                com = self.slicer.plane0.center_of_mass()
                camera = {
                    "pos": com - self.orientation * 2 * np.linalg.norm(com),
                    "viewup": (0, -1, 0),
                    "clipping_range": (19531, 40903),
                }
        if kwargs.get("export_html"):
            self.scene.export(kwargs.get("export_html"))
        elif kwargs.get("export_glb"):
            export_glb_path = kwargs.get("export_glb")
            if export_glb_path is not None:
                path = Path(export_glb_path)
                # TODO exclude_root option
                # exclude_root = kwargs.get("exclude_root", True)
                export_powerpoint_compatible_glb(self, path)
        else:
            self.scene.render(
                camera=camera, interactive=self.interactive, zoom=self.zoom
            )
        return self.scene

    def plot(
        self,
        show_legend: bool = False,
        xlabel: str = "µm",
        ylabel: str = "µm",
        hide_axes: bool = False,
        filename: Optional[str] = None,
        cbar_label: Optional[str] = None,
        show_cbar: bool = True,
        **kwargs,
    ) -> plt.Figure:
        """
        Plots the heatmap in 2D using matplotlib.

        This method generates a 2D visualization of the heatmap data in
        a standalone matplotlib figure.

        Parameters
        ----------
        show_legend : bool, optional
            If True, displays a legend for the plotted regions.
            Default is False.
        xlabel : str, optional
            Label for the x-axis. Default is "µm".
        ylabel : str, optional
            Label for the y-axis. Default is "µm".
        hide_axes : bool, optional
            If True, hides the axes for a cleaner look. Default is False.
        filename : Optional[str], optional
            Path to save the figure to. If None, the figure is not saved.
            Default is None.
        cbar_label : Optional[str], optional
            Label for the colorbar. If None, no label is displayed.
            Default is None.
        show_cbar : bool, optional
            If True, displays a colorbar alongside the subplot.
            Default is True.
        **kwargs : dict
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        plt.Figure
            The matplotlib figure object for the plot.

        Notes
        -----
        This method is used to generate a standalone plot of
        the heatmap data.
        """

        f, ax = plt.subplots(figsize=(10, 8))
        f, ax = self.plot_subplot(
            fig=f,
            ax=ax,
            show_legend=show_legend,
            xlabel=xlabel,
            ylabel=ylabel,
            hide_axes=hide_axes,
            cbar_label=cbar_label,
            show_cbar=show_cbar,
            **kwargs,
        )
        if self.tight_layout_2d:
            f.tight_layout()

        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()
        return f

    def plot_subplot(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        show_legend: bool = False,
        xlabel: str = "µm",
        ylabel: str = "µm",
        hide_axes: bool = False,
        cbar_label: Optional[str] = None,
        show_cbar: bool = True,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots a heatmap in a subplot within a given figure and axes.

        This method is responsible for plotting a single subplot within a
        larger figure, allowing for the creation of complex multi-plot
        visualizations.

        Parameters
        ----------
        fig : plt.Figure, optional
            The figure object in which the subplot is plotted.
        ax : plt.Axes, optional
            The axes object in which the subplot is plotted.
        show_legend : bool, optional
            If True, displays a legend for the plotted regions.
            Default is False.
        xlabel : str, optional
            Label for the x-axis. Default is "µm".
        ylabel : str, optional
            Label for the y-axis. Default is "µm".
        hide_axes : bool, optional
            If True, hides the axes for a cleaner look. Default is False.
        cbar_label : Optional[str], optional
            Label for the colorbar. If None, no label is displayed.
            Default is None.
        show_cbar : bool, optional
            Display a colorbar alongside the subplot. Default is True.
        **kwargs : dict
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        plt.Figure, plt.Axes
            A tuple containing the figure and axes objects used for the plot.

        Notes
        -----
        This method modifies the provided figure and axes objects in-place.
        """
        projected, _ = self.slicer.get_structures_slice_coords(
            self.regions_meshes, self.scene.root
        )

        segments: List[Dict[str, Union[str, np.ndarray, float]]] = []
        for r, coords in projected.items():
            name, segment_nr = r.split("_segment_")
            x = coords[:, 0]
            y = coords[:, 1]
            # calculate area of polygon with Shoelace formula
            area = 0.5 * np.abs(
                np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
            )

            segments.append(
                {
                    "name": name,
                    "segment_nr": segment_nr,
                    "coords": coords,
                    "area": area,
                }
            )

        # Sort region segments by area (largest first)
        segments.sort(key=lambda s: s["area"], reverse=True)
        # for annotate_less_clutter
        track_segment_ocurr = set()
        for segment in segments:
            name = segment["name"]
            segment_nr = segment["segment_nr"]
            coords = segment["coords"]

            ax.fill(
                coords[:, 0],
                coords[:, 1],
                color=self.colors[name],
                label=name if segment_nr == "0" and name != "root" else None,
                lw=1,
                ec="k",
                zorder=-1 if name == "root" else None,
                alpha=0.3 if name == "root" else None,
            )

            display_text = self.get_region_annotation_text(name)
            if display_text is not None and name not in track_segment_ocurr:
                track_segment_ocurr.add(name)
                annotation_pos = find_annotation_position_inside_polygon(
                    coords
                )
                if annotation_pos is not None:
                    ax.annotate(
                        display_text,
                        xy=annotation_pos,
                        ha="center",
                        va="center",
                        **(
                            self.annotate_text_options
                            if self.annotate_text_options is not None
                            else {}
                        ),
                    )

        if show_cbar:
            # make colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            # cmap = mpl.cm.cool
            norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
            if self.label_regions is True:
                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(
                        norm=None,
                        cmap=mpl.cm.get_cmap(self.cmap, len(self.values)),
                    ),
                    cax=cax,
                )
            else:
                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=self.cmap), cax=cax
                )

            if cbar_label is not None:
                cbar.set_label(cbar_label)

            if self.label_regions is True:
                cbar.ax.set_yticklabels(
                    [r.strip() for r in self.values.keys()]
                )

        # style axes
        ax.invert_yaxis()
        ax.axis("equal")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.set(title=self.title)

        if isinstance(self.orientation, str) or np.sum(self.orientation) == 1:
            # orthogonal projection
            ax.set(xlabel=xlabel, ylabel=ylabel)

        if hide_axes:
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set(xlabel="", ylabel="")

        if show_legend:
            ax.legend()

        return fig, ax
