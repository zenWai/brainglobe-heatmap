# BG Heatmaps
Rendering heatmaps with brainrender.

`Bgheatmaps` makes it easier to create beautiful heatmaps from anatomical data mapping a scalar values for each brain region (e.g., number of labelled cells in each region) to color and creating beautiful visualizations in 3D and 2D.

![](images/hm_2d.png)
![](images/hm_3d.png)

### installation
`pip install bgheatmaps`


## User guide
The starting point for a heatmap visualization is a `dict` assigning scalar values to a set of brain regions (identified by their acronym).
For example:

```python 
    regions = dict(  # scalar values for each region
        TH=1,
        RSP=0.2,
        AI=0.4,
        SS=-3,
        MO=2.6,
        ...
    )
```

`bgheatmaps` creates a brainrender 3D `Scene` with the given regions colored according the values in the dictionary.
Next, to create visualizations like the ones shown above, the three dimensional sceen needs to be **sliced** to expose
the relevant parts.
This is done by specifying the position and orientation of a `Plane` which cuts through the scene.

![](images/planning_1.png)

The orientation is set by the direction of a *normal vector* specified by the user.

![](images/planning_2.png)

Everything that is on the side opposite where the normal vector will be cut and discarded.
To keep a section of the 3D brain, two planes with normal vectors facing in opposite directions are used:

![](images/planning_3.png)

and everything in-between the two planes is kept as a slice.

### Slicing plane position
Finding the right position and orientation to the plane can take some tweaking. `Bgheatmaps` provides a `planner` class that makes the process easier by showing the position of the planes and how they intersect with the user provided regions (see image above).
In `examples/plan.py` there's an example showing how to use the `planner`:

```python
import bgheatmaps as bgh


planner = bgh.plan(
    regions,
    position=(
        8000,
        5000,
        5000,
    ),  
    orientation="frontal",  # orientation, or 'sagittal', or 'top' or a tuple (x,y,z)
    thickness=2000,  # thickness of the slices used for rendering (in microns)
)
```

The position of the center of the plane is given by a set of `(x, y, z)` coordinates. The orientation can be specified by a string (`frontal`, `sagittal`, `top`) which will result in a standard orthogonal slice, or by a vector `(x, y, z)` with the orientation along the 3 axes.

### Visualization
Once happy with the position of the slicing planes, creating a visualization is as simple as:

```python

bgh.heatmap(
    values,
    position=(
        8000,
        5000,
        5000,
    ),  # displacement along the AP axis relative to midpoint
    orientation="top",  # 'frontal' or 'sagittal', or 'top' or a tuple (x,y,z)
    title="top view",
    vmin=-5,
    vmax=3,
    cmap='Red',
    format="2D",
).show()
```

Here, `format` spcifies if a 2D plot should be made (using `matplotlib`) or a 3D rendering instead (using `brainrender`). The `cmap` parameter specifies the colormap used and `vmin, vmax` the color range.

### Regions coordinates
You can use `bgheatmaps` to get the coordinates of the 2D 'slices' (in the 2D plane's coordinates system):


```python

regions = ['TH', 'RSP', 'AI', 'SS', 'MO', 'PVZ', 'LZ', 'VIS', 'AUD', 'RHP', 'STR', 'CB', 'FRP', 'HIP', 'PA']


coordinates = bgh.get_plane_coordinates(
    regions,
    position=(
        8000,
        5000,
        5000,
    ),  # displacement along the AP axis relative to midpoint
    orientation="frontal",  # 'frontal' or 'sagittal', or 'top' or a tuple (x,y,z)
)
```

