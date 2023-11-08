import numpy as np
import pyvista as pv
from stl import mesh

# Load STL files
file1 = 'C:/Users/Keshav Verma/Desktop/Work/HIWI GAPP/Graph_Model_Project/NOV_07/test/Assem1 - Part2-2-1.STL'
file2 = 'C:/Users/Keshav Verma/Desktop/Work/HIWI GAPP/Graph_Model_Project/NOV_07/test/Assem1 - Part1-2-1.STL'

# Load meshes
mesh1 = mesh.Mesh.from_file(file1)
mesh2 = mesh.Mesh.from_file(file2)

# Reshape the points array
points1 = mesh1.points.reshape((-1, 3))
points2 = mesh2.points.reshape((-1, 3))

# Convert vectors to integer array-like format
vectors1 = mesh1.vectors.reshape((-1, 3)).astype(int)
vectors2 = mesh2.vectors.reshape((-1, 3)).astype(int)

# Define threshold for coincidence
coincidence_threshold = 0.001  # Adjust as needed

# Perform comparison
contact_points = []
bounded_points = []
free_points = []

for i in range(len(points1)):
    is_bounded = False
    for j in range(len(points2)):
        # Check if points are coincident
        dist = np.linalg.norm(points1[i] - points2[j])
        if dist < coincidence_threshold:
            contact_points.append(points1[i])
            is_bounded = True
            break
    if is_bounded:
        bounded_points.append(points1[i])
    else:
        free_points.append(points1[i])

# Create PyVista meshes
pv_mesh1 = pv.PolyData(points1, vectors1)
pv_mesh2 = pv.PolyData(points2, vectors2)

# Create PyVista point cloud for contact points
contact_points = np.array(contact_points)
bounded_points = np.array(bounded_points)
free_points = np.array(free_points)

pv_contact_points = pv.PolyData(contact_points)
pv_bounded_points = pv.PolyData(bounded_points)
pv_free_points = pv.PolyData(free_points)

# Create the first plotting window for contact points
plotter1 = pv.Plotter()

# Add the mesh to the first plotter
plotter1.add_mesh(pv_mesh1, color='blue', opacity=0.5)

# Add the contact points as a point cloud in the first plotter
plotter1.add_points(pv_contact_points, color='green', point_size=5)

# Show the first plotter with contact points
plotter1.show()
# Create the second plotting window for bounded points
plotter2 = pv.Plotter()

# Add the mesh to the second plotter
plotter2.add_mesh(pv_mesh2, color='red', opacity=0.5)

# Add the bounded points as a point cloud in the second plotter
plotter2.add_points(pv_bounded_points, color='yellow', point_size=5)
# Show the second plotter with bounded points
plotter2.show()
