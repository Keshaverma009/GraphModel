import os
import numpy as np
import pyvista as pv
from stl import mesh
from scipy.spatial import cKDTree

class MeshAnalyzer:
    def __init__(self, folder_path, coincidence_threshold=0.1):
        self.folder_path = folder_path
        self.coincidence_threshold = coincidence_threshold

    def load_mesh(self, file_path):
        mesh_data = mesh.Mesh.from_file(file_path)
        vertices = mesh_data.vectors.reshape(-1, 3)
        return vertices

    def find_contact_points(self, vertices_1, vertices_2):
        kdtree_1 = cKDTree(vertices_1)
        kdtree_2 = cKDTree(vertices_2)
        contact_points = []
        for idx_1, point_1 in enumerate(vertices_1):
            coincident_points = kdtree_2.query_ball_point(point_1, r=self.coincidence_threshold)
            if len(coincident_points) > 0:
                for idx_2 in coincident_points:
                    contact_points.append(point_1)
                    contact_points.append(vertices_2[idx_2])
        return np.array(contact_points).reshape(-1, 3)

    def visualize_meshes(self, vertices_1, vertices_2, contact_points):
        pv_contact_points = pv.PolyData(contact_points)
        part_mesh_1 = pv.PolyData(vertices_1)
        part_mesh_2 = pv.PolyData(vertices_2)

        plotter = pv.Plotter()
        plotter.add_mesh(part_mesh_1, color='red', opacity=0.5, name='Part 1')
        plotter.add_mesh(part_mesh_2, color='blue', opacity=0.5, name='Part 2')
        plotter.add_points(pv_contact_points, color='green', point_size=5)
        plotter.show()

    def process_files(self):
        stl_files = [f for f in os.listdir(self.folder_path) if f.endswith('.STL')]
        for i in range(len(stl_files)):
            for j in range(i + 1, len(stl_files)):
                file_name_1 = stl_files[i]
                file_name_2 = stl_files[j]
                file_path_1 = os.path.join(self.folder_path, file_name_1)
                file_path_2 = os.path.join(self.folder_path, file_name_2)

                vertices_1 = self.load_mesh(file_path_1)
                vertices_2 = self.load_mesh(file_path_2)
                
                contact_points = self.find_contact_points(vertices_1, vertices_2)

                if len(contact_points) > 0:
                    self.visualize_meshes(vertices_1, vertices_2, contact_points)

# Usage
folder_path = 'C:/Users/Keshav Verma/Desktop/Work/HIWI GAPP/Graph_Model_Project/NOV_07/test/'
analyzer = MeshAnalyzer(folder_path)
analyzer.process_files()
