import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon   # used to draw polygon shapes
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap   # used for transparent colors

# ------------------------------------------------------------
# This script defines the four court polygons using coordinates
# that were manually collected earlier (via Find_Coordinates.py).
#
# It then:
#   1. Plots the polygons for visual verification
#   2. Saves the polygon vertex data into zones.pkl
#
# zones.pkl is what your YOLO pipeline loads to determine
# which court each detected person belongs to.
# ------------------------------------------------------------

n_pt = 10  # total number of coordinate points (not actually used)

# ------------------------------------------------------------
# objp[] contains ALL the manually collected coordinates.
# These represent key points around the courts in the camera view.
#
# Each tuple is (x, y) in ORIGINAL image coordinates.
# ------------------------------------------------------------
objp = np.array([
    (0, 0),        # 0  (not used)
    (1498, 145),   # 1
    (1774, 172),   # 2
    (1663, 241),   # 3
    (1263, 161),   # 4
    (1005, 455),   # 5
    (977, 167),    # 6
    (691, 153),    # 7
    (250, 230),    # 8
    (152, 157),    # 9
    (438, 133)     # 10
], dtype=np.float32)

# ------------------------------------------------------------
# Define each court polygon using 4 corner points.
#
# The order of points matters — they must form a proper quadrilateral.
#
# Court 1 = right-most court
# Court 4 = left-most court
#
# These polygons match the physical layout of the SDC courts.
# ------------------------------------------------------------

# Court 1 (Right-most court)
court1 = Polygon([objp[1], objp[2], objp[3], objp[4]], closed=True)

# Court 2 (Middle-right court)
court2 = Polygon([objp[3], objp[4], objp[6], objp[5]], closed=True)

# Court 3 (Middle-left court)
court3 = Polygon([objp[5], objp[6], objp[7], objp[8]], closed=True)

# Court 4 (Left-most court)
court4 = Polygon([objp[7], objp[8], objp[9], objp[10]], closed=True)

# Collect polygons for plotting
polygons = [court1, court2, court3, court4]

# ------------------------------------------------------------
# Plot the polygons so you can visually confirm they look correct.
# This is ONLY for debugging — not used in production.
# ------------------------------------------------------------
fig, ax = plt.subplots()

pc = PatchCollection(
    polygons,
    facecolor=['red', 'orange', 'yellow', 'green', 'blue'],  # colors for each court
    alpha=0.4,                                               # transparency
    edgecolor='black'
)

ax.add_collection(pc)

# Set axis limits to match the camera resolution
ax.set_xlim(0, 2000)
ax.set_ylim(0, 1000)

# Keep aspect ratio correct so courts aren't stretched
ax.set_aspect('equal', adjustable='box')

plt.title('Court Polygons')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# IMPORTANT:
# Invert Y-axis because OpenCV uses (0,0) at top-left,
# but matplotlib uses (0,0) at bottom-left.
plt.gca().invert_yaxis()

plt.show()

# ------------------------------------------------------------
# Save polygon vertex data into zones.pkl
#
# This file is loaded by your YOLO pipeline to determine
# which court each detected person belongs to.
#
# Each polygon is saved as a list of (x, y) coordinates.
# ------------------------------------------------------------
import pickle

zones_save = {
    "Right-most court": court1.get_verts(),
    "Middle right court": court2.get_verts(),
    "Middle left court": court3.get_xy(),
    "Left-most court": court4.get_xy(),
}

with open("zones.pkl", "wb") as f:
    pickle.dump(zones_save, f)
