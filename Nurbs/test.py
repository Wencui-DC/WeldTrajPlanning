# Importing NURBS module
from geomdl import NURBS
from geomdl import utilities
# Importing visualization module
from geomdl.visualization import VisMPL as vis
import numpy as np

# Creating a curve instance
crv = NURBS.Curve()

# Make a quadratic curve
crv.degree = 3

#######################################################
# Skipping control points and knot vector assignments #
#######################################################
crv.ctrlpts = np.array([
    [0, 1],
    [1, 3],
    [4, 2],
    [6, 0],
    [7, 4]
])

crv.knotvector = utilities.generate_knot_vector(crv.degree, crv.ctrlpts_size)

# Set the visualization component and render the curve
crv.vis = vis.VisCurve2D()
crv.render()



