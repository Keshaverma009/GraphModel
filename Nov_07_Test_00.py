import os
import math
import numpy as np
import pyvista as pv
import trimesh
import fcl
import matplotlib.pyplot as plt  # Import Matplotlib for saving images
import vtk
class GraphGenerator:

    def __init__(self):

        self.distance_precision = 0.01
        self.angle_precision = 1.0
        self.volume_precision = 0.01
        self.displacement_vector_num = 14

        sqrt_3 = math.sqrt(3)

        self.displacement_vectors = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                                              [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                                              [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                                              [1.0 / sqrt_3, 1.0 / sqrt_3, 1.0 / sqrt_3],
                                              [-1.0 / sqrt_3, -1.0 / sqrt_3, -1.0 / sqrt_3],
                                              [1.0 / sqrt_3, 1.0 / sqrt_3, -1.0 / sqrt_3],
                                              [-1.0 / sqrt_3, -1.0 / sqrt_3, 1.0 / sqrt_3],
                                              [-1.0 / sqrt_3, 1.0 / sqrt_3, 1.0 / sqrt_3],
                                              [1.0 / sqrt_3, -1.0 / sqrt_3, -1.0 / sqrt_3],
                                              [-1.0 / sqrt_3, 1.0 / sqrt_3, -1.0 / sqrt_3],
                                              [1.0 / sqrt_3, -1.0 / sqrt_3, 1.0 / sqrt_3]])

        self.displacement_vectors *= self.distance_precision

        self.limit_distance = 50000.0

        self.file_folder_path = None
        self.file_names = None
        self.file_num = 0

        self.relationship_matrix = None

    def classify_parts_contact_relationships(self, folder_path):
        
        file_names = os.listdir(folder_path)
        file_names.sort()

        self.file_folder_path = folder_path
        self.file_names = file_names
        self.file_num = len(self.file_names)
        self.relationship_matrix = np.zeros((self.file_num, self.file_num))

        for i in range(0, self.file_num - 1):
            model1_path = os.path.join(self.file_folder_path, self.file_names[i])
            model1 = pv.read(model1_path)

            j = i + 1
            while j < self.file_num:
                model2_path = os.path.join(self.file_folder_path, self.file_names[j])
                model2 = pv.read(model2_path)

                contact = self.classify_contact_relationship_by_collision(model1, model2)
                blocking = self.classify_blocking_relationship(model1, model2)
                free = self.classify_free_relationship(model1, model2)

                if contact:
                    print(f"Contact: {self.file_names[i]} and {self.file_names[j]}")
                    if contact[1] is not None:
                        self.visualize_contact_area(contact[1], model1_path, model2_path)
                elif blocking:
                    print(f"Blocking: {self.file_names[i]} and {self.file_names[j]}")
                elif free:
                    print(f"Free: {self.file_names[i]} and {self.file_names[j]}")

                j += 1
    
    def classify_contact_relationship(self, model1, model2):
        return self.classify_contact_relationship_by_collision(model1, model2)
    
    def classify_contact_relationship_by_collision(self, model1, model2):
        contact_polydata = None  # Initialize contact_polydata to None
        
        model2_trimesh = trimesh.Trimesh(model2.points, model2.faces.reshape((model2.n_faces, 4))[:, 1:])
        model2_bvh = trimesh.collision.mesh_to_BVH(model2_trimesh)
        model2_collision_object = fcl.CollisionObject(model2_bvh, fcl.Transform())

        for i in range(0, self.displacement_vector_num):
            moved_model1 = model1.copy(deep=True)
            moved_model1.translate(self.displacement_vectors[i, :], inplace=True)

            contact_area = self.compute_contact_area(moved_model1, model2_collision_object)

            if contact_area > 0:
                contact_polydata = vtk.vtkPolyData()
                contact_points = vtk.vtkPoints()
                contact_cells = vtk.vtkCellArray()

                contact_points.InsertNextPoint(self.displacement_vectors[i, :])
                contact_cells.InsertNextCell(1)
                contact_cells.InsertCellPoint(contact_points.GetNumberOfPoints() - 1)

                contact_polydata.SetPoints(contact_points)
                contact_polydata.SetVerts(contact_cells)
                
                # Return True and polydata when there is a contact
                return True, contact_polydata

        # Return False and None when there is no contact
        return False, None
    
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
    
    def classify_blocking_relationship(self, model1, model2):
        # Check if model1 completely encloses model2
        if model1.volume > model2.volume and self.is_model_inside(model1, model2):
            return True
        
        # Check if model2 completely encloses model1
        if model2.volume > model1.volume and self.is_model_inside(model2, model1):
            return True
        
        # Additional criteria based on your specific requirements can be added here
        
        return False

    def is_model_inside(self, container_model, enclosed_model):
        enclosed_model_trimesh = trimesh.Trimesh(enclosed_model.points, enclosed_model.faces.reshape((enclosed_model.n_faces, 4))[:, 1:])
        container_model_trimesh = trimesh.Trimesh(container_model.points, container_model.faces.reshape((container_model.n_faces, 4))[:, 1:])

        container_model_bvh = trimesh.collision.mesh_to_BVH(container_model_trimesh)
        enclosed_model_bvh = trimesh.collision.mesh_to_BVH(enclosed_model_trimesh)

        container_collision_object = fcl.CollisionObject(container_model_bvh, fcl.Transform())
        enclosed_collision_object = fcl.CollisionObject(enclosed_model_bvh, fcl.Transform())

        collision_request = fcl.CollisionRequest()
        collision_result = fcl.CollisionResult()

        fcl.collide(container_collision_object, enclosed_collision_object, collision_request, collision_result)
        return not collision_result.is_collision

    def classify_free_relationship(self, model1, model2):
        # Check if model1 and model2 are not in contact
        if not self.classify_contact_relationship(model1, model2):
            # Check if model1 and model2 are not blocking each other
            if not self.classify_blocking_relationship(model1, model2):
                return True
        
        return False  
    
    def compute_contact_area(self, moved_model1, model2_collision_object):
        contact_areas = []  # List to store individual contact areas

        moved_model1 = trimesh.Trimesh(moved_model1.points,
                                    moved_model1.faces.reshape((moved_model1.n_faces, 4))[:, 1:])
        moved_model1_bvh = trimesh.collision.mesh_to_BVH(moved_model1)
        moved_model1_collision_object = fcl.CollisionObject(moved_model1_bvh, fcl.Transform())

        collision_request = fcl.CollisionRequest(num_max_contacts=1000)
        collision_result = fcl.CollisionResult()

        fcl.collide(moved_model1_collision_object, model2_collision_object, collision_request, collision_result)

        for contact in collision_result.contacts:
            contact_area_polydata = vtk.vtkPolyData()
            contact_points = vtk.vtkPoints()
            contact_cells = vtk.vtkCellArray()

            # Extract contact point from the result
            contact_point = contact.pos

            # Add the contact point to polydata
            contact_points.InsertNextPoint(contact_point)
            contact_cells.InsertNextCell(1)
            contact_cells.InsertCellPoint(contact_points.GetNumberOfPoints() - 1)

            contact_area_polydata.SetPoints(contact_points)
            contact_area_polydata.SetVerts(contact_cells)

            contact_areas.append(contact_area_polydata)

        return contact_areas
    
    def classify_contact_relationship_by_collision(self, model1, model2):
        contact_areas = []  # Initialize contact areas list
        
        model2_trimesh = trimesh.Trimesh(model2.points, model2.faces.reshape((model2.n_faces, 4))[:, 1:])
        model2_bvh = trimesh.collision.mesh_to_BVH(model2_trimesh)
        model2_collision_object = fcl.CollisionObject(model2_bvh, fcl.Transform())

        for i in range(0, self.displacement_vector_num):
            moved_model1 = model1.copy(deep=True)
            moved_model1.translate(self.displacement_vectors[i, :], inplace=True)

            contact_area = self.compute_contact_area(moved_model1, model2_collision_object)

            if contact_area:
                contact_areas.extend(contact_area)  # Extend the list with individual contact areas

        if contact_areas:
            # Return True and the list of contact areas when there is a contact
            return True, contact_areas

        # Return False and an empty list when there is no contact
        return False, []
    
    def visualize_contact_area(self, contact_areas, model1_path, model2_path):
        # Load STL models
        model1 = pv.read(model1_path)
        model2 = pv.read(model2_path)

        # Create a PyVista plotter
        plotter = pv.Plotter()

        # Add models to the plotter
        plotter.add_mesh(model1, color='red', opacity=0.5, label='Model 1')
        plotter.add_mesh(model2, color='blue', opacity=0.5, label='Model 2')

        # Visualize individual contact areas using polydata
        for contact_area in contact_areas:
            plotter.add_mesh(contact_area, color='green', point_size=10, render_points_as_spheres=True, label='Contact Area')

        # Set up the plotter and show the visualization
        plotter.show()
        
if __name__ == "__main__":
    generator = GraphGenerator()
    folder_path = "C:/Users/Keshav Verma/Desktop/Work/HIWI GAPP/Graph_Model_Project/NOV_07/test/"
    generator.classify_parts_contact_relationships(folder_path)
