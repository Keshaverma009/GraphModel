from stl import mesh
from scipy.spatial.transform import Rotation as R
import numpy as np

# Load STL files
mesh1 = mesh.Mesh.from_file('C:/Users/Keshav Verma/Desktop/Work/HIWI GAPP/Graph_Model_Project/Nov_07/test/Assem1 - Part1-2-1.STL')
mesh2 = mesh.Mesh.from_file('C:/Users/Keshav Verma/Desktop/Work/HIWI GAPP/Graph_Model_Project/Nov_07/test/Assem1 - Part4-2-1.STL')

# Calculate centroids for mesh1 and mesh2
centroid1 = mesh1.points.mean(axis=0)
centroid2 = mesh2.points.mean(axis=0)

# Calculate joint translation vector between centroids of mesh1 and mesh2
joint_translation = centroid2 - centroid1

# Calculate joint rotation matrix (rotation from mesh1 to mesh2)
mesh1_normal = np.cross(mesh1.normals[0], mesh1.normals[1])
mesh2_normal = np.cross(mesh2.normals[0], mesh2.normals[1])

# Check if the normals are valid and non-zero
if np.linalg.norm(mesh1_normal) > 0 and np.linalg.norm(mesh2_normal) > 0:
    joint_rotation_axis = np.cross(mesh1_normal, mesh2_normal)
    joint_rotation_angle = np.arccos(np.dot(mesh1_normal, mesh2_normal) / (np.linalg.norm(mesh1_normal) * np.linalg.norm(mesh2_normal)))
    
    # Check if joint rotation angle is valid (non-zero)
    if joint_rotation_angle > 0:
        joint_rotation_matrix = R.from_rotvec(joint_rotation_angle * joint_rotation_axis)
        rotation_valid = True
    else:
        rotation_valid = False
else:
    rotation_valid = False

# Initialize DOF matrices for mesh1 and mesh2 as 3x4 matrices
dof_matrix_mesh1 = np.zeros((3, 4))
dof_matrix_mesh2 = np.zeros((3, 4))

# Check which translation components are free or blocked for mesh1
for i in range(3):
    if joint_translation[i] != 0:
        dof_matrix_mesh1[i, i] = 1  # Translation component is free

# Check if the rotation is valid, and set rotational DOF for mesh1 accordingly
if rotation_valid:
    dof_matrix_mesh1[:, 3] = 1  # All rotation components are free

# Check which translation components are free or blocked for mesh2
for i in range(3):
    if joint_translation[i] != 0:
        dof_matrix_mesh2[i, i] = 1  # Translation component is free

# Check if the rotation is valid, and set rotational DOF for mesh2 accordingly
if rotation_valid:
    dof_matrix_mesh2[:, 3] = 1  # All rotation components are free

# Print the DOF matrices for mesh1 and mesh2
print("DOF Matrix for Mesh 1:")
print(dof_matrix_mesh1)
print("DOF Matrix for Mesh 2:")
print(dof_matrix_mesh2)
