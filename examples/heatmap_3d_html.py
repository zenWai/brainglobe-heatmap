import brainglobe_heatmap as bgh

values = dict(  # scalar values for each region
    TH=1,
    RSP=0.2,
    AI=0.4,
    SS=-3,
    MO=2.6,
    PVZ=-4,
    LZ=-3,
    VIS=2,
    AUD=0.3,
    RHP=-0.2,
    STR=0.5,
    CB=0.5,
    FRP=-1.7,
    HIP=3,
    PA=-4,
)

# TODO: position 0 and thickness 100000 to export to html
scene = bgh.Heatmap(
    values,
    position=0,
    orientation="frontal",  # or 'sagittal', or 'horizontal' or a tuple (x,y,z)
    thickness=100000,
    title="frontal",
    vmin=-5,
    vmax=3,
    format="3D",
)
scene.show(export_html="myfile.html")
