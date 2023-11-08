import os
import vtk
import numpy as np
import trimesh
import fcl
from scipy.spatial import cKDTree

def extract_points_and_faces(polydata):
    cells = polydata.GetPolys()
    cells.InitTraversal()
    face = vtk.vtkIdList()

    vertices = []
    faces = []

    while cells.GetNextCell(face):
        face_vertices = []
        for i in range(face.GetNumberOfIds()):
            point_id = face.GetId(i)
            point = polydata.GetPoints().GetPoint(point_id)
            face_vertices.append(point)
        if len(face_vertices) == 3:
            faces.append(face_vertices)
        elif len(face_vertices) == 4:
            # Convert quads to triangles for trimesh
            faces.append([face_vertices[0], face_vertices[1], face_vertices[2]])
            faces.append([face_vertices[0], face_vertices[2], face_vertices[3]])
        else:
            print(f"Skipping non-triangular/non-quadrilateral face with {len(face_vertices)} vertices")
    vertices = np.array(polydata.GetPoints().GetData())
    faces = np.array(faces)
    return vertices, faces

class MeshAnalyzer:

    def __init__(self, folder_path, coincidence_threshold=0.2):
        self.folder_path = folder_path
        self.coincidence_threshold = coincidence_threshold

    def load_mesh(self, file_path):
        mesh_data = vtk.vtkSTLReader()
        mesh_data.SetFileName(file_path)
        mesh_data.Update()
        polydata = mesh_data.GetOutput()
        return extract_points_and_faces(polydata)
    
    def detect_collision(self, moved_model1, model2_collision_object):

        moved_model1 = trimesh.Trimesh(moved_model1.points,
                                       moved_model1.faces.reshape((moved_model1.n_faces, 4))[:, 1:])

        moved_model1_bvh = trimesh.collision.mesh_to_BVH(moved_model1)

        moved_model1_collision_object = fcl.CollisionObject(moved_model1_bvh, fcl.Transform())

        collision_request = fcl.CollisionRequest()
        collision_result = fcl.CollisionResult()

        collision_information = fcl.collide(moved_model1_collision_object, model2_collision_object,
                                            collision_request, collision_result)

        return collision_result.is_collision
    
    def find_contact_points(self, vertices_1, faces_1, vertices_2, faces_2):
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

    def calculate_joint_coordinates(self, contact_points):
        if len(contact_points) > 0:
            joint_coordinates = np.mean(contact_points, axis=0)
            return joint_coordinates
        else:
            return None

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

                vertices_1, faces_1 = self.load_mesh(file_path_1)
                vertices_2, faces_2 = self.load_mesh(file_path_2)

                contact_points = self.find_contact_points(vertices_1, faces_1, vertices_2, faces_2)

                # Create Trimesh objects for collision detection
                moved_model1 = trimesh.Trimesh(vertices_1, faces_1)
                model2_collision_object = fcl.BVHModel()
                model2_collision_object.beginModel(len(vertices_2), len(faces_2))
                model2_collision_object.addSubModel(vertices_2, faces_2)
                model2_collision_object.endModel()

                if len(contact_points) > 0:
                    print(f"Contact found between {file_name_1} and {file_name_2}")
                    joint_coordinates = self.calculate_joint_coordinates(contact_points)
                    if joint_coordinates is not None:
                        print(f"Joint Coordinates: {joint_coordinates}")
                    self.visualize_meshes_with_contact_points(moved_model1, vertices_2, contact_points)
                else:
                    if self.detect_collision(moved_model1, model2_collision_object):
                        print(f"Collision detected between {file_name_1} and {file_name_2}")
                    else:
                        print(f"No contact or collision detected between {file_name_1} and {file_name_2}")

    # Rest of your code...



# Usage
folder_path = 'C:/Users/Keshav Verma/Desktop/Work/HIWI GAPP/Graph_Model_Project/NOV_07/test/'
analyzer = MeshAnalyzer(folder_path)
analyzer.process_files()