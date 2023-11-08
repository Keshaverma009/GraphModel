import os
import math
import numpy as np
import trimesh
import fcl

class DOFGenerator:

    def __init__(self):
        self.distance_precision = 10
        self.angle_precision = 1.0
        self.volume_precision = 0.01
        self.rotation_angle = math.radians(1.0)  # 1 degree rotation

        self.file_folder_path = None
        self.file_names = None
        self.file_num = 0

        self.dof_matrices_model1 = []
        self.dof_matrices_model2 = []

    def compute_dof_matrices(self, folder_path):

        if not folder_path.endswith("/"):
            folder_path += "/"

        file_names = os.listdir(folder_path)
        file_names.sort()

        self.file_folder_path = folder_path
        self.file_names = file_names

        self.file_num = len(self.file_names)

        for i in range(0, self.file_num - 1):

            model1 = trimesh.load_mesh(self.file_folder_path + self.file_names[i])

            j = i + 1

            while j < self.file_num:

                print(f"{i}   {j}   {self.file_names[i]}   {self.file_names[j]}")

                model2 = trimesh.load_mesh(self.file_folder_path + self.file_names[j])

                # Compute DOF matrices for model1 and model2 using the optimal joint coordinates
                dof_matrix_model1 = self.compute_dof_matrix(model1, model2)
                dof_matrix_model2 = self.compute_dof_matrix(model2, model1)

                self.dof_matrices_model1.append(dof_matrix_model1)
                self.dof_matrices_model2.append(dof_matrix_model2)

                j += 1

    def compute_dof_matrix(self, model, other_model):
        dof_matrix = np.zeros((3, 2), dtype=int)

        for i in range(2):
            # Translate in the positive direction along X and Y
            for j in range(2):
                moved_model = model.copy()
                translation_vector = [0, 0, 0]
                translation_vector[i] = self.distance_precision if j == 0 else -self.distance_precision
                moved_model.apply_translation(translation_vector)
                if not self.detect_collision(moved_model, other_model):
                    dof_matrix[i, j] = 1

        # Rotate in the positive and negative direction along Z
        for i in range(2):
            moved_model = model.copy()
            rotation_angle = self.rotation_angle if i == 0 else -self.rotation_angle
            rotated_vertices = np.dot(moved_model.vertices - moved_model.center_mass, 
                                      np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                                                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                                                [0, 0, 1]]).T)
            moved_model.vertices = rotated_vertices + moved_model.center_mass
            if not self.detect_collision(moved_model, other_model):
                dof_matrix[2, i] = 1

        return dof_matrix
    
    def detect_collision(self, moved_model, other_model):

        moved_model_trimesh = trimesh.Trimesh(moved_model.vertices, moved_model.faces)
        other_model_trimesh = trimesh.Trimesh(other_model.vertices, other_model.faces)

        moved_model_bvh = trimesh.collision.mesh_to_BVH(moved_model_trimesh)
        other_model_bvh = trimesh.collision.mesh_to_BVH(other_model_trimesh)

        moved_model_collision_object = fcl.CollisionObject(moved_model_bvh, fcl.Transform())
        other_model_collision_object = fcl.CollisionObject(other_model_bvh, fcl.Transform())

        collision_request = fcl.CollisionRequest()
        collision_result = fcl.CollisionResult()

        fcl.collide(moved_model_collision_object, other_model_collision_object,
                    collision_request, collision_result)

        return collision_result.is_collision

# Example usage:
if __name__ == "__main__":
    generator = DOFGenerator()
    # Compute DOF matrices and detect collisions
    generator.compute_dof_matrices(folder_path="C:/Users/Keshav Verma/Desktop/Work/HIWI GAPP/Graph_Model_Project/Sept_18_Test/STL for Matrix")

    # Access DOF matrices for model1 and model2
    dof_matrices_model1 = generator.dof_matrices_model1
    dof_matrices_model2 = generator.dof_matrices_model2

    for i, (dof_matrix1, dof_matrix2) in enumerate(zip(dof_matrices_model1, dof_matrices_model2)):
        if i == 0:  # Check if it's the first model pair
            print(f"DOF Matrices for Model Pair {i + 1}:")
            print("DOF Matrix for Model 1:")
            print(dof_matrix1)
