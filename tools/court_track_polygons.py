import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon   # used to plot patches
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap   # used to plot transparent

n_pt= 10 #total number of points

# All the points from the coordinates spreadsheet
objp = np.array([
(0, 0), # 0
(1498, 145), # 1
(1774, 172), # 2
(1663, 241), # 3
(1263, 161), # 4
(1005, 455), # 5
(977, 167), # 6
(691, 153), # 7
(250, 230), # 8
(152, 157), # 9
(438, 133)  # 10
], dtype=np.float32)

# Make the the polygons 

# Court 1 (Right-most court)
court1 = Polygon([objp[1], objp[2], objp[3], objp[4]], closed=True)
# Court 2 (Middle right court)
court2 = Polygon([objp[3], objp[4], objp[6], objp[5]], closed=True)
# Court 3 (Middle left court)
court3 = Polygon([objp[5], objp[6], objp[7], objp[8]], closed=True)
# Court 4 (Left-most court)
court4 = Polygon([objp[7], objp[8], objp[9], objp[10]], closed=True)

polygons = [court1, court2, court3, court4]
# Plotting
fig, ax = plt.subplots()
pc = PatchCollection(
    polygons,
    facecolor=['red', 'orange', 'yellow', 'green', 'blue'],
    alpha=0.4,
    edgecolor='black'
)
ax.add_collection(pc)
ax.set_xlim(0, 2000)
ax.set_ylim(0, 1000)
ax.set_aspect('equal', adjustable='box')
plt.title('Court Polygons')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
plt.show()


#save zones data
import pickle

zones_save = {
    "Right-most court": court1.get_verts(),
    "Middle right court": court2.get_verts(),
    "Middle left court": court3.get_xy(),
    "Left-most court": court4.get_xy(),
}

with open("zones.pkl", "wb") as f:
    pickle.dump(zones_save, f)