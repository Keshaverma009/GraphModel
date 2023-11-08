import os
import numpy as np
import vtk
from scipy.spatial import cKDTree

class MeshAnalyzer:
    def __init__(self, folder_path, coincidence_threshold=0.2):
        self.folder_path = folder_path
        self.coincidence_threshold = coincidence_threshold

    def load_mesh(self, file_path):
        mesh_data = vtk.vtkSTLReader()
        mesh_data.SetFileName(file_path)
        mesh_data.Update()
        return mesh_data.GetOutput()

    def find_contact_points(self, mesh_1, mesh_2):
        vertices_1 = np.array(mesh_1.GetPoints().GetData())
        vertices_2 = np.array(mesh_2.GetPoints().GetData())

        # Define threshold for coincidence
        coincidence_threshold = 0.5  # Adjust as needed

        contact_points = []
        bounded_points = []
        free_points = []

        for i in range(len(vertices_1)):
            is_bounded = False
            for j in range(len(vertices_2)):
                # Check if points are coincident
                dist = np.linalg.norm(vertices_1[i] - vertices_2[j])
                if dist < coincidence_threshold:
                    contact_points.append(vertices_1[i])
                    is_bounded = True
                    break
            if is_bounded:
                bounded_points.append(vertices_1[i])
            else:
                free_points.append(vertices_1[i])

        # Print the bounded and free points if needed
        # print("Bounded Points:", bounded_points)
        # print("Free Points:", free_points)

        return np.array(contact_points).reshape(-1, 3)

    def visualize_meshes_with_contact_points(self, mesh_1, mesh_2, contact_points):
        mapper_1 = vtk.vtkPolyDataMapper()
        mapper_1.SetInputData(mesh_1)
        actor_1 = vtk.vtkActor()
        actor_1.SetMapper(mapper_1)

        mapper_2 = vtk.vtkPolyDataMapper()
        mapper_2.SetInputData(mesh_2)
        actor_2 = vtk.vtkActor()
        actor_2.SetMapper(mapper_2)

        points = vtk.vtkPoints()
        vertices = np.array(contact_points)
        for vertex in vertices:
            points.InsertNextPoint(vertex)

        contact_point_polydata = vtk.vtkPolyData()
        contact_point_polydata.SetPoints(points)

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(0.1)
        glyph_mapper = vtk.vtkGlyph3D()
        glyph_mapper.SetInputData(contact_point_polydata)
        glyph_mapper.SetSourceConnection(sphere.GetOutputPort())

        contact_point_mapper = vtk.vtkPolyDataMapper()
        contact_point_mapper.SetInputConnection(glyph_mapper.GetOutputPort())

        contact_point_actor = vtk.vtkActor()
        contact_point_actor.SetMapper(contact_point_mapper)
        contact_point_actor.GetProperty().SetColor(0.0, 1.0, 0.0)

        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor_1)
        renderer.AddActor(actor_2)
        renderer.AddActor(contact_point_actor)

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)

        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        render_window.SetSize(800, 600)
        renderer.SetBackground(1.0, 1.0, 1.0)
        render_window.Render()
        render_window_interactor.Start()

    def process_files(self):
        stl_files = [f for f in os.listdir(self.folder_path) if f.endswith('.STL')]
        
        for i in range(len(stl_files)):
            for j in range(i + 1, len(stl_files)):
                file_name_1 = stl_files[i]
                file_name_2 = stl_files[j]
                file_path_1 = os.path.join(self.folder_path, file_name_1)
                file_path_2 = os.path.join(self.folder_path, file_name_2)

                mesh_1 = self.load_mesh(file_path_1)
                mesh_2 = self.load_mesh(file_path_2)

                contact_points = self.find_contact_points(mesh_1, mesh_2)

                if len(contact_points) > 0:
                    print(f"Contact found between {file_name_1} and {file_name_2}")
                    self.visualize_meshes_with_contact_points(mesh_1, mesh_2, contact_points)

# Usage
folder_path = 'C:/Users/Keshav Verma/Desktop/Work/HIWI GAPP/Graph_Model_Project/NOV_07/test/'
analyzer = MeshAnalyzer(folder_path)
analyzer.process_files()